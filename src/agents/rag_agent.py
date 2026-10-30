from anyio import run
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document

from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware

from core.resources import open_resources, Resources
from agents.contracts import LoreAnswer

from typing import List, Tuple, Optional

import asyncio
import logging


logger = logging.getLogger(__name__)


# ============================================= PARAMETERS =============================================

MODEL_NAME = "nvidia/nemotron-3.5-lightning:free"
FALLBACK_NAME = "xiaomi/mimo-v2.6-flash"

CANDIDATES = 50
TOP_K = 20
RERANK_TOP = 10


# ============================================= PROMPTS =============================================

GRADE_INSTRUCTION = """Decide whether the chunk in input contains information that help anwer the query.
Answer 'yes' if the passage is useful to answer the query, otherwise 'no'
"""

GRADE_CRITERIA = {
    "yes": "the chunk in input contains information that help answer the query",
    "no": "the passage is off-topic or does not help answer the question."
}

SYSTEM_PROMPT = """Your are an helpuful assistant.
You have to answer questions about SIKTC: Something Is Killing The Children and its spin-offs from BOOM! Studios 
using the wiki passaged returned by the search tool.

For each search, provide:
- answer: the answer to the question pose by the user. DO NOT INCLUDE ANY reasoning.
- retrieved: the URLs of the passages you used for answering the question
- confidence: how confident you are about the answer. Use 'high' if you are confident about the answer, 
'unsure' if the answer requires more information, and 'low' if no passage are returned or they are off-topics.
The 'answer' MUST BE 2-4 sentences, about 250 words long.


Follow these rules:
1. Always call the search_lore tool before answering. The wiki is in English, so search in English, even when the 
question is in another language.
2. If the first results are incomplete or your confidence is 'unsure', search again with a different query. You can 
rewrite the query by providing other names, narrowing the topic, etc. You can also provide a short answer if you think
it could improve the search. DO AT MOST 3 searches. 
3. Use only fact from wiki passages to answer the question. DO NOT use your knowledge about the comic series. DO NOT invent anything.
4. If earch_lore returns NO_RESULTS or nothing useful, say that the wiki does not cover it and set confidence to low.
5. Write the answer in English
"""


# ============================================= FUNCTIONS =============================================


async def score_with_jev(resources: Resources, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:

    async def grade(document: str) -> str:
        
        state = f"<query>{query}</query>\n<chunk>{document}</chunk>\n"
        return await resources.evaluator.choice(state, GRADE_INSTRUCTION, GRADE_CRITERIA)
        
    verdicts = await asyncio.gather(
        *(grade(d.page_content or "") for d in documents), 
        return_exceptions=True
    )

    to_kept = []
    for verdict, document in zip(verdicts, documents):

        if isinstance(verdict, Exception):
            logger.warning("LDM failed for chunk %r: %r", document.id, document.page_content)
            to_kept.append(document) # save for safety, LLM will prune

        if verdict.choice.strip().lower() == "yes":
            to_kept.append(document)

    return to_kept


async def rag_function(resources: Resources, 
                       question: str) -> List[Document]:

    [question_vector] = await resources.model.get_embeddings([question]) # estrae un solo vettore
    chunks: List[Tuple[Document, float]] = await resources.retriever.search_chunks_hybrid(
            index_name=resources.index_name,
            query_vector=question_vector,
            query_text=question, 
            candidates=CANDIDATES,
            top_k=TOP_K
    )

    reranked_chunks = await resources.model.rerank(question, [c for c, _ in chunks], top_n=RERANK_TOP)
    
    return (await score_with_jev(resources, question, [c for c, _ in reranked_chunks]))


def build_search_lore_tool(resources: Resources):

    @tool("search_lore", description="Retrieve relevant passages to answer the question")
    async def search_lore(query: str, runtime: ToolRuntime):
        docs = await rag_function(resources, query) # aggiungere qui l'oggetto del runtime
        return format_chunks(docs)

    return search_lore


def format_chunks(docs: List[Document]) -> List[str]:

    if not docs:
        return "NO RESULTS: nothing relevant found with this query."
    
    blocks = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        infobox = "\n".join([f"{k}: {v}" for k, v in meta.items() if "infobox" in k])
        blocks.append(
            f"[{i}] {meta.get("title", "?")} | {meta.get("section_path", "")}\n"
            f"source: {meta.get("source_url", "")}\n"
            f"metadata:\n{infobox}\n"
            f"passage:\n{doc.page_content}"
        )
    
    return "\n\n".join(blocks)


# ============================================= AGENT =============================================


def build_lore_subagent(resources: Resources):

    chat_model = init_chat_model(
        model=MODEL_NAME,
        model_provider="openrouter",
        reasoning=None
    )

    fallback_model = init_chat_model(
        model=FALLBACK_NAME,
        model_provider="openrouter",
        reasoning=None
    )

    return create_agent(
        model=chat_model,
        system_prompt=SYSTEM_PROMPT,
        response_format=LoreAnswer,
        tools=[build_search_lore_tool(resources)],
        middleware=[
            ToolCallLimitMiddleware(tool_name="search_lore", run_limit=3, exit_behavior="continue"),
            ModelCallLimitMiddleware(run_limit=6), # 3 per search_lore, 1 in caso di errore, 1 per LoreAnswer, 1 per errore
            ModelFallbackMiddleware(fallback_model),
            ModelRetryMiddleware(max_retries=2, on_failure="error"),
        ],
        name="lore_agent"
    )


# ============================================= TESTING =============================================

async def main():

    from langchain.messages import HumanMessage
    from pprint import pprint

    async with open_resources() as resources:

        agent = build_lore_subagent(resources)

        question = "Tell me about the butcher's war?"
        messages = [
            HumanMessage(content=question)
        ]

        response = await agent.ainvoke({"messages": messages})

        pprint(response["structured_response"])


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    asyncio.run(main())

    """
    ATTENDED REPLY

    LoreAnswer(

    answer="The Butcher's War is the fifth and final story arc of the *House of Slaughter* comic series, 
    a spin-off of *Something Is Killing the Children* (SIKTC) published by BOOM! Studios. 
    It collects issues #21–25, written by Tate Brombal with art by Antonio Fuso, and serves as the conclusion 
    to the Butcher trilogy (which includes the preceding volumes: The Butcher's Mark, Scarlet, The Butcher's Return, and Alabaster). 
    The arc was collected into a trade paperback titled *House of Slaughter – Volume Five: The Butcher's War*, 
    released on November 19, 2024 (ISBN 9781637969816).", 
    
    retrieved=["https://something-is-killing-the-children.fandom.com/wiki/The_Butcher's_War_(comic_arc)", 
    'https://something-is-killing-the-children.fandom.com/wiki/Comics_collections'], 
    
    confidence='high')

    """
