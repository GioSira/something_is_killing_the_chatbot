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
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from pinecone import DenseVectorQuery, PineconeAsyncio, TextQuery
from pinecone.exceptions import NotFoundException
from pinecone.schema_builder import SchemaBuilder

from retrieval.utils import _reciprocal_rank_fusion
from retrieval.utils import to_pinecone_record
from retrieval.errors import IndexNotFoundError, EmbeddingDimensionMismatchError


logger = logging.getLogger(__name__)

VECTOR_FIELD = "embedding"
TEXT_FIELDS = ("page_content", "summary")
CHUNKS_NS = "chunks"
SUMMARIES_NS = "summaries"



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
                                   candidates: int = 50,
                                   top_k: int = 20,
                                   filter: Optional[Dict[str, Any]] = None,
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
                fields.pop(key, None)

            content = fields.pop("page_content", "")
            documents.append((Document(id=hit.id, page_content=content, metadata=fields), score))

        return documents
