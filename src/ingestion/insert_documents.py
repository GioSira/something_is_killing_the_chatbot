from pathlib import Path
import ujson
import logging
import asyncio
import time
from typing import Any, Dict


from ingestion.summary import get_embeddings
from retrieval.pincone_retrieval import PineconeRetriever, CHUNKS_NS, SUMMARIES_NS


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        force=True)



async def insert(vector_size : int = 1024):

    async with PineconeRetriever() as retriever:
        await retriever.create_index("siktc", dimension=vector_size)
        await retriever.insert_records("siktc", CHUNKS_NS, records, vector_key="text_embedding")
        await retriever.insert_records("siktc", SUMMARIES_NS, records, vector_key="summary_embedding")

async def retrieve(query: str, filter: Dict[str, Any] = None):
    
    async with PineconeRetriever() as retriever:
        query_embedding = get_embeddings([query])[0]
        retrieved_docs = await retriever.search_chunks_hybrid(
            index_name="siktc", 
            query_text=query,
            query_vector=query_embedding,
            top_k=5
        )

        for doc, score in retrieved_docs:
            text = doc.get("page_content", "")
            summary = doc.get("summary", "")
            print(f"score: {score}\ntext:{text}\nsummary:{summary}\n\n")
        


# PATH TO COLLECTION
root = Path(__file__).resolve().parents[2]
path = root / "collection" / "siktc_with_summaries.jsonl"

# LOAD JSON FILE
with open(path, encoding="utf-8") as f:
    records = [ujson.loads(line) for line in f if line.strip()]

dimension = len(records[0]["text_embedding"])
logger.info("%d record, embedding di dimensione %d", len(records), dimension)

# INGESTION
start_time = time.time()
asyncio.run(insert(vector_size=dimension))
end_time = time.time()
logger.info("time: %f", end_time - start_time)

# CHECKING
query = "Who is Octo?"
asyncio.run(retrieve(query))
