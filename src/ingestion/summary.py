from langchain_core.documents import Document
from openrouter import OpenRouter
from openrouter.utils.retries import BackoffStrategy, RetryConfig
from openrouter.errors import (TooManyRequestsResponseError, BadGatewayResponseError,
                               ProviderOverloadedResponseError, ServiceUnavailableResponseError,
                               RequestTimeoutResponseError)


import os
import json
import time
import logging
from dotenv import load_dotenv

from typing import Any, Dict, List
from itertools import groupby
from pathlib import Path

from ingestion.chunk import infobox_text, load_markdown
from utils.retry_decorators import retry_sync

load_dotenv()


logger = logging.getLogger(__name__)


MODEL_NAME = "scaledown/summarize"
EMBED_TYPE = "liquid/lfm-2.5-embedding-350m:free"

retry_config = RetryConfig(
    "backoff",
    BackoffStrategy(
        initial_interval=500,
        max_interval=10_000,
        exponent=1.5,
        max_elapsed_time=int(60 * 1000),
    ),
    retry_connection_errors=True
)

PARAMS = {
    "retries": retry_config,
    "timeout_ms": 100000,
    # output stabile e non creativo
    "temperature": 0.0,
    "max_completion_tokens": 2_000,
    "reasoning": {
        "effort": "none"
    }
}

XCEPTIONS = (
    TooManyRequestsResponseError,
    BadGatewayResponseError,
    ProviderOverloadedResponseError,
    ServiceUnavailableResponseError,
    RequestTimeoutResponseError,
)

# Metadati del chunk che aiutano l'LLM a capire il contesto. Il resto (id, token,
# licenza, url...) è rumore nel prompt.
PROMPT_METADATA_KEYS = ("title", "doc_type", "section_path", "series", "issue_min", "issue_max")


SYSTEM_PROMPT = f"""You are a helpful assistant. 
You write retrieval summaries for chunks of a wiki about the comic series
Something Is Killing the Children (BOOM! Studios) and its spin-offs. Each summary
is indexed next to its chunk to help a search system find the chunk from a user's question.

You receive the full document, the chunk's metadata and the chunk.
Write 2-4 sentences, at most 80 words, in English, as plain text.

What to cover:
- Story events or scenes: who is involved, what happens, where, and why, when stated.
- Characters: their role and the story arcs they appear in.
- Locations: what the place is and what happens there.
- Story arcs: the main conflict, places and characters.
- Tables or infobox data: what the table lists (e.g. "release dates of each issue
  of the arc") and the key entities; do not copy the rows.
- Real-world topics (publication, creators, TV or film adaptations): what is covered,
  including names of creators, issues and dates.

Rules:
1. Use the full document only to identify who or what the chunk refers to:
   replace pronouns and vague references ("she", "the order", "the creature") with full names.
2. State only facts found in the chunk. Do not add events from other parts of the
   document, especially later in the story: the summary must not reveal more than the chunk.
3. For story content, leave out issue numbers and creator credits; they are in the metadata.
4. Do not invent details. If something is not stated, leave it out.
"""

openrouter_client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"), retry_config=retry_config)


class EmptyResponseError(Exception):
    """Il modello ha risposto senza testo (es. token esauriti nel ragionamento)."""


def _prompt_metadata(metadata: Dict[str, Any]) -> str:
    return "\n".join(f"{key}: {metadata[key]}" for key in PROMPT_METADATA_KEYS
                     if metadata.get(key) not in (None, ""))


@retry_sync(XCEPTIONS + (EmptyResponseError,), max_retries=5, base_delay=2.0, max_delay=60.0,
            factor=2.0, jitter=0.25)
def generate_summary(chunk_text: str, document_text: str, chunk_metadata: Dict[str, Any]) -> str:

    # Ordine pensato per la cache dei prompt: system e documento sono uguali per tutti
    # i chunk dello stesso file, quindi vanno prima; metadati e chunk cambiano, vanno dopo.
    llm_input = (
        f"<document>\n{document_text}\n</document>\n\n"
        f"<metadata>\n{_prompt_metadata(chunk_metadata)}\n</metadata>\n\n"
        f"<chunk>\n{chunk_text}\n</chunk>"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": llm_input}
    ]

    response = openrouter_client.chat.send(
        messages=messages,
        model=MODEL_NAME,
        **PARAMS
    )

    if not (content := (response.choices[0].message.content or "").strip()):
        raise EmptyResponseError(f"Risposta vuota da {MODEL_NAME}")
    
    logger.info("Generated summary for %r: %r", chunk_metadata["chunk_id"], content)

    return content


@retry_sync(XCEPTIONS, max_retries=5, base_delay=2.0, max_delay=60.0, factor=2.0, jitter=0.25)
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Più testi in una sola chiamata (l'API embeddings accetta una lista in input)."""

    response = openrouter_client.embeddings.generate(
        model=EMBED_TYPE,
        input=texts,
        retries=PARAMS["retries"],
        timeout_ms=PARAMS["timeout_ms"]
    )

    logger.info("Generated embeddings for %r", texts)

    return [[float(x) for x in item.embedding] for item in response.data]


def load_document(source_path: str) -> str:
    """Testo completo del documento: infobox (che sta nel frontmatter) + corpo."""

    front, body = load_markdown(Path(source_path))
    infobox = front.get("infobox")
    header = f"Infobox\n{infobox_text(infobox)}\n\n" if infobox else ""

    return header + body.strip()


def _done_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()

    with output_path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def process_documents(documents: List[Document], output_path: str | Path) -> int:
    """Riassume ed embedda i chunk, scrivendo un record JSONL per chunk appena è pronto.

    Se il processo si interrompe, rilanciandolo riparte dai chunk mancanti.
    Ritorna il numero di chunk elaborati in questa esecuzione.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(output_path)

    todo = [d for d in documents if d.metadata["chunk_id"] not in done]
    todo.sort(key=lambda d: d.metadata["source_path"])

    processed = 0
    with output_path.open("a", encoding="utf-8") as out:
        for source_path, group in groupby(todo, key=lambda d: d.metadata["source_path"]):

            document = load_document(source_path)
            logger.info("Loaded document %r", source_path)

            for chunk in group:
                logger.info("Processing chunk %r", chunk.metadata["chunk_id"])
                summary = generate_summary(chunk.page_content, document, chunk.metadata)
                time.sleep(0.3)
                summary_embedding, text_embedding = get_embeddings([summary, chunk.page_content])
                time.sleep(0.3)

                record = {
                    "id": chunk.metadata["chunk_id"],
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "summary": summary,
                    "summary_embedding": summary_embedding,
                    "text_embedding": text_embedding,
                }
                logger.info("Created record for chunk %r", chunk.metadata["chunk_id"])

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                logger.info("Wrote record for chunk %r", chunk.metadata["chunk_id"])

                processed += 1

    return processed


if __name__ == "__main__":

    from ingestion.chunk import chunk_folder

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,   # sovrascrive configurazioni fatte prima da librerie importate
    )

    chunk_docs = chunk_folder("collection/siktc")
    try:
        n = process_documents(chunk_docs, "collection/siktc_with_summaries.jsonl")
        print(f"Completato: {n} chunk elaborati in questa esecuzione")
    except XCEPTIONS as exc:
        done = len(_done_ids(Path("collection/siktc_with_summaries.jsonl")))
        print(f"Limite del provider raggiunto ({type(exc).__name__}). "
              f"Salvati {done}/{len(chunk_docs)} chunk: rilancia più tardi per continuare.")
