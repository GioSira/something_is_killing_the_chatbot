from dataclasses import dataclass

from retrieval.pincone_retrieval import PineconeRetriever
from llm.llm_openrouter import EmbedAndRerankModel
from llm.jev_openrouter import JevLoader

from typing import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager


INDEX_NAME = "siktc"
EMBED_MODEL = "liquid/lfm-2.5-embedding-350m:free"
RERANK_MODEL = "voyageai/rerank-3"


@dataclass(frozen=True)
class Resources:

    retriever: PineconeRetriever
    model: EmbedAndRerankModel
    evaluator: JevLoader

    index_name: str


@asynccontextmanager
async def open_resources() -> AsyncIterator[Resources]:
    async with AsyncExitStack() as stack:
        retriever = await stack.enter_async_context(PineconeRetriever())
        model = await stack.enter_async_context(EmbedAndRerankModel(rerank_model=RERANK_MODEL, embed_model=EMBED_MODEL))
        evaluator = await stack.enter_async_context(JevLoader())
        yield Resources(retriever=retriever, model=model, evaluator=evaluator, index_name=INDEX_NAME)
