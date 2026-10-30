from openrouter import OpenRouter
from openrouter.utils.retries import BackoffStrategy, RetryConfig
from openrouter.errors import (TooManyRequestsResponseError, BadGatewayResponseError,
                               ProviderOverloadedResponseError, ServiceUnavailableResponseError,
                               RequestTimeoutResponseError)

from utils.retry_decorators import retry_async

from langchain_core.documents import Document

from dotenv import load_dotenv
from typing import List, Tuple
import os


load_dotenv()

XCEPTIONS = (
    TooManyRequestsResponseError,
    BadGatewayResponseError,
    ProviderOverloadedResponseError,
    ServiceUnavailableResponseError,
    RequestTimeoutResponseError,
)


# ====================================== MODEL FOR EMBEDDINGS AND RERANKING ===============================

class EmbedAndRerankModel:
    
    def __init__(self, 
                rerank_model: str,
                embed_model: str,

        ):

        self.retry_config = RetryConfig(
            "backoff",
            BackoffStrategy(
                initial_interval=500,
                max_interval=10_000,
                exponent=1.5,
                max_elapsed_time=int(60 * 1000),
            ),
            retry_connection_errors=True
        )

        self.client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"), retry_config=self.retry_config)
        self.rerank_model = rerank_model
        self.embed_model = embed_model


    async def aclose(self) -> None:
        close = getattr(self.client, "__aexit__", None)
        if close is not None:
            await close(None, None, None)
 
 
    async def __aenter__(self) -> "EmbedAndRerankModel":
        return self
 
 
    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose()
    
    # =============================== EMBED AND RERANK ===========================================

    
    @retry_async(XCEPTIONS, max_retries=5, base_delay=2.0, max_delay=60.0, factor=2.0, jitter=0.25)
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Più testi in una sola chiamata (l'API embeddings accetta una lista in input)."""

        response = await self.client.embeddings.generate_async(
            model=self.embed_model,
            input=texts,
            timeout_ms=120000,
            retries=self.retry_config
        )

        return [[float(x) for x in item.embedding] for item in response.data]

    
    @retry_async(XCEPTIONS, max_retries=5, base_delay=5)
    async def rerank(self, query: str, documents: List[Document], top_n: int = 10) -> List[Tuple[Document, float]]:

        if top_n < 1:
            raise ValueError(f"top_n deve essere almeno 1, ricevuto {top_n}")
        
        if not documents:
            return []

        document_text = [doc.page_content or "" for doc in documents]

        response = await self.client.rerank.rerank_async(
            model=self.rerank_model,
            documents=document_text, 
            query=query, 
            top_n=top_n,
            retries=self.retry_config,
            timeout_ms=120000
        )

        ranked = sorted(response.results, key=lambda r: r.relevance_score, reverse=True)

        return [(documents[r.index], r.relevance_score) for r in ranked]
