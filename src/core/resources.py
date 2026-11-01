from dataclasses import dataclass

from retrieval.pincone_retrieval import PineconeRetriever
from llm.llm_openrouter import EmbedAndRerankModel
from llm.jev_openrouter import JevLoader

from typing import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from tavily.async_tavily import AsyncTavilyClient
from pinecone import PineconeAsyncio

import os
import httpx
from dotenv import load_dotenv

load_dotenv()



INDEX_NAME = "siktc"
EMBED_MODEL = "liquid/lfm-2.5-embedding-350m:free"
RERANK_MODEL = "voyageai/rerank-3"

METRON_URL = "https://metron.cloud/api/"
METRON_KEY = os.getenv("METRON_KEY")



@dataclass(frozen=True)
class Resources:

    retriever: PineconeRetriever
    model: EmbedAndRerankModel
    evaluator: JevLoader
    
    tavily: AsyncTavilyClient
    metron: httpx.AsyncClient

    index_name: str


@asynccontextmanager
async def open_resources() -> AsyncIterator[Resources]:
    async with AsyncExitStack() as stack:
        retriever = await stack.enter_async_context(PineconeRetriever())
        model = await stack.enter_async_context(EmbedAndRerankModel(rerank_model=RERANK_MODEL, embed_model=EMBED_MODEL))
        evaluator = await stack.enter_async_context(JevLoader())
        tavily = await stack.enter_async_context(AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY")))

        metron_client = await stack.enter_async_context(httpx.AsyncClient(
                base_url=METRON_URL,
                headers={
                    "Authorization": f"Bearer {METRON_KEY}"
                },
                timeout=30.0,
                follow_redirects=True,
        ))
        
        yield Resources(retriever=retriever, 
                        model=model, 
                        evaluator=evaluator, 
                        tavily=tavily,
                        metron=metron_client,
                        index_name=INDEX_NAME
        )
