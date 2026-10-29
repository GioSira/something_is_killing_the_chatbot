"""Indice Pinecone (API a schema, SDK pinecone >= 10) per i chunk di siktc_with_summaries.jsonl.

Uno schema Pinecone ammette al massimo UN campo dense_vector. Per tenere entrambi i
vettori che calcoli (testo del chunk e riassunto) usiamo due namespace con lo stesso schema:

    namespace "chunks"     -> embedding = text_embedding    (+ ricerca full-text BM25)
    namespace "summaries"  -> embedding = summary_embedding

In ricerca: una query vettoriale per namespace + una query full-text su "chunks",
fuse con Reciprocal Rank Fusion sull'id del chunk.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from pinecone import DenseVectorQuery, PineconeAsyncio, TextQuery
from pinecone.exceptions import NotFoundException
from pinecone.schema_builder import SchemaBuilder
from retrieval.utils import _reciprocal_rank_fusion

logger = logging.getLogger(__name__)

VECTOR_FIELD = "embedding"
TEXT_FIELDS = ("page_content", "summary")
CHUNKS_NS = "chunks"
SUMMARIES_NS = "summaries"
UNKNOWN_ISSUE = -1          # pagine senza range di uscite (es. molti personaggi)

_FIELD_NAME = re.compile(r"[^0-9A-Za-z_]")


class IndexNotFoundError(Exception):
    """Errore sollevato quando un indice richiesto non esiste in Pinecone."""


class EmbeddingDimensionMismatchError(Exception):
    """Errore sollevato quando le dimensioni dell'embedding non coincidono con quelle dell'indice."""


# ============================== RECORD ==============================

def _field_value(value: Any) -> Any:
    """Pinecone accetta solo stringhe, numeri, booleani e liste di stringhe."""
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = json.loads(value)          # es. categories salvate come JSON
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed                       # lista vera: filtrabile con $in
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    return None


def to_pinecone_record(record: Dict[str, Any], vector_key: str) -> Dict[str, Any]:
    """Record del JSONL -> documento Pinecone: metadati al primo livello, nomi di campo puliti."""
    fields: Dict[str, Any] = {}
    for key, value in record.get("metadata", {}).items():
        value = _field_value(value)
        if value not in (None, "", []):
            # "infobox.written_by" -> "infobox_written_by": il punto nei nomi dei campi
            # è ambiguo nei filtri
            fields[_FIELD_NAME.sub("_", key)] = value

    # Sentinella per lo spoiler guard: un filtro su issue_max escluderebbe i documenti
    # senza il campo, quindi tutte le pagine senza range.
    fields.setdefault("issue_min", UNKNOWN_ISSUE)
    fields.setdefault("issue_max", UNKNOWN_ISSUE)

    return {
        **fields,
        "_id": record["id"],
        "page_content": record["page_content"],
        "summary": record.get("summary", ""),
        VECTOR_FIELD: [float(x) for x in record[vector_key]],
    }


def spoiler_filter(progress: Dict[str, int], strict: bool = True,
                   allow_unknown: bool = True) -> Optional[Dict[str, Any]]:
    """Filtro Pinecone per lo spoiler guard.

    progress: {"Something Is Killing the Children": 20, "House of Slaughter": 5}
    strict=True: il chunk passa solo se l'utente ha finito il suo arco (issue_max).
    """
    bound = "issue_max" if strict else "issue_min"
    clauses: List[Dict[str, Any]] = [
        {"$and": [{"series": {"$eq": series}},
                  {bound: {"$lte": last}},
                  {"issue_min": {"$gte": 0}}]}
        for series, last in progress.items()
    ]
    if allow_unknown:
        clauses.append({"issue_min": {"$eq": UNKNOWN_ISSUE}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$or": clauses}


# ============================== RETRIEVER ==============================

class PineconeRetriever:

    def __init__(self, cloud: str = "aws",
                 region: str = "us-east-1",
                 metric: str = "cosine",
                 client: Optional[PineconeAsyncio] = None):

        self._cloud = cloud
        self._region = region
        self._metric = metric
        self._client = client or self._connect()
        self._indexes: Dict[str, Any] = {}      # un client per indice, riusato
        self._lock = asyncio.Lock()

    # ------------------------------ lifetime ------------------------------

    async def aclose(self) -> None:
        for index in self._indexes.values():
            await index.close()
        self._indexes.clear()
        await self._client.close()

    async def __aenter__(self) -> "PineconeRetriever":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose()

    @staticmethod
    def _connect() -> PineconeAsyncio:
        load_dotenv()   # legge PINECONE_API_KEY dal file .env
        return PineconeAsyncio()

    # ------------------------------ indexes ------------------------------

    async def create_index(self, index_name: str, dimension: int, language: str = "en") -> bool:
        """Crea l'indice se non esiste (la chiamata attende che sia pronto).

        Se esiste già, controlla che la dimensione coincida con quella dei tuoi embedding.
        """
        if await self._client.has_index(index_name):
            existing = await self._index_dimension(index_name)
            if existing is not None and existing != dimension:
                raise EmbeddingDimensionMismatchError(
                    f"l'indice {index_name!r} ha dimensione {existing}, gli embedding {dimension}: "
                    f"cancellalo o usa un altro nome")

            return False

        schema = (SchemaBuilder()
                  .add_string_field(name="page_content", full_text_search={}, language=language, stemming=True, stop_words=True)
                  .add_string_field(name="summary", full_text_search={}, language=language, stemming=True, stop_words=True)
                  .add_dense_vector_field(name=VECTOR_FIELD, dimension=dimension, metric=self._metric)
                  .build())

        await self._client.indexes.create(
            name=index_name,
            schema=schema,
            deployment={"deployment_type": "managed", "cloud": self._cloud, "region": self._region},
        )

        logger.info("Creato l'indice %r (dimensione %d)", index_name, dimension)

        return True

    async def _index_dimension(self, index_name: str) -> Optional[int]:
        desc = await self._client.describe_index(index_name)
        schema = desc.schema.to_dict() if getattr(desc, "schema", None) else {}
        field = schema.get("fields", {}).get(VECTOR_FIELD, {})

        return field.get("dimension") or getattr(desc, "dimension", None)

    async def _index(self, index_name: str):
        # Il lock evita che due richieste insieme aprano due client per lo stesso indice.
        async with self._lock:
            if index_name not in self._indexes:
                if not await self._client.has_index(index_name):
                    raise IndexNotFoundError(f"l'indice {index_name!r} non esiste")

                host = (await self._client.describe_index(index_name)).host
                self._indexes[index_name] = self._client.IndexAsyncio(host=host)

            return self._indexes[index_name]

    # ------------------------------ insert ------------------------------

    async def insert_records(self, index_name: str, namespace: str,
                             records: Sequence[Dict[str, Any]], vector_key: str) -> int:

        """Upsert dei record del JSONL usando il vettore `vector_key`. Ritorna quanti sono andati a buon fine."""
        if not records:
            return 0

        index = await self._index(index_name)
        documents = [to_pinecone_record(r, vector_key) for r in records]

        # batch_upsert divide in batch, gestisce il limite di dimensione (413) e la concorrenza.
        result = await index.documents.batch_upsert(namespace=namespace, documents=documents,
                                                     batch_size=50, show_progress=True)

        if result.has_errors:
            logger.error("Upsert su %r/%r: %d record falliti. Primi errori: %s",
                         index_name, namespace, result.failed_item_count, result.errors[:3])

        logger.info("Upsert su %r/%r: %d/%d record", index_name, namespace,
                    result.successful_item_count, len(documents))

        return result.successful_item_count


    async def delete_page(self, index_name: str, namespace: str, pageid: int) -> bool:
        """Cancella tutti i chunk di una pagina della wiki (es. prima di reindicizzarla)."""
        try:
            index = await self._index(index_name)
            await index.documents.delete(namespace=namespace, filter={"pageid": {"$eq": pageid}})
        except (IndexNotFoundError, NotFoundException):
            return False
        return True

    # ------------------------------ retrieval ------------------------------

    async def search_chunks_hybrid(self, index_name: str,
                                   query_vector: Sequence[float],
                                   query_text: str,
                                   top_k: int = 10,
                                   filter: Optional[Dict[str, Any]] = None,
                                   candidates: int = 50,
                                   rrf_k: int = 60,
                                   dense_namespaces: Sequence[str] = (CHUNKS_NS, SUMMARIES_NS),
                                   text_namespace: str = CHUNKS_NS) -> List[Tuple[Document, float]]:
        
        if top_k < 1:
            raise ValueError("top_k deve essere almeno 1")

        index = await self._index(index_name)
        common = dict(top_k=max(candidates, top_k), filter=filter, include_fields=["*"])
        vector = [float(x) for x in query_vector]

        searches = [
            index.documents.search(namespace=ns, **common,
                                   score_by=[DenseVectorQuery(field=VECTOR_FIELD, values=vector)])
            for ns in dense_namespaces
        ]

        if query_text and query_text.strip():
            searches.append(index.documents.search(
                namespace=text_namespace, **common,
                score_by=[TextQuery(query=query_text, fields=list(TEXT_FIELDS))]))

        # Le richieste partono insieme: la latenza è quella della più lenta.
        responses = await asyncio.gather(*searches)
        fused = _reciprocal_rank_fusion([r.matches for r in responses], k=rrf_k)
        return self._hits_to_documents(fused[:top_k])

    @staticmethod
    def _hits_to_documents(hits: Sequence[Tuple[Any, float]]) -> List[Tuple[Document, float]]:
        documents: List[Tuple[Document, float]] = []
        for hit, score in hits:
            fields = hit.to_dict()
            for key in ("_id", "_score", "id", "score", VECTOR_FIELD):
                fields.pop(key, None)            # il vettore non serve a valle
            content = fields.pop("page_content", "")
            documents.append((Document(id=hit.id, page_content=content, metadata=fields), score))
        return documents


# ============================== MAIN ==============================

async def main() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    path = root / "collection" / "siktc_with_summaries.jsonl"
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    dimension = len(records[0]["text_embedding"])
    logger.info("%d record, embedding di dimensione %d", len(records), dimension)

    async with PineconeRetriever() as retriever:
        #await retriever.create_index("siktc", dimension=dimension)
        #await retriever.insert_records("siktc", CHUNKS_NS, records, vector_key="text_embedding")
        #await retriever.insert_records("siktc", SUMMARIES_NS, records, vector_key="summary_embedding")
        matches = await retriever.search_chunks_hybrid("siktc", query_vector=[0.1]*1024, query_text="Erica")
        print(matches)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        force=True)

    asyncio.run(main())   # un solo event loop per tutto: il client asincrono è legato al loop