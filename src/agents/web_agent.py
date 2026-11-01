from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import (ModelRetryMiddleware, 
                                        ModelFallbackMiddleware, 
                                        ToolCallLimitMiddleware,
                                        ModelCallLimitMiddleware)

from core.resources import Resources
from agents.contracts import LoreAnswer

from typing import List, Dict, Any, Tuple
import logging
import json

logger = logging.getLogger(__name__)

# ============================================= PARAMETERS =============================================

MODEL_NAME = "nvidia/nemotron-3.5-lightning:free"
FALLBACK_NAME = "xiaomi/mimo-v2.6-flash"

# ============================================= PROMPTS =============================================

SYSTEM_PROMPT = """You are a helpful assistant. 
Your task is to asnwer questions about SIKTC: Something is Killing the Children comic series from Boom! Studios
and its spin-offs (House of Slaughter, Fall of the House of Slaughter, etc.) only using the content returned 
by the websearch tool. 

For each search, provide:
- 'answer': the answer to the question posed by the user. DO NOT include any reasoning.
- 'retrieved': the URLs of the web pages used to answer the question.
- 'confidence': how confident you are about the answer. Reply with 'high' if the documents provide a clear answer, 
'unsure' if they provide a partial answer, and 'low' if they are off-topics, they provide a very limited answer or no answer.
The 'answer' MUST BE 2-4 sentences, about 250 words long.

Follow these rules:
1. Always call the websearch tool before providing an answer.
2. Always search in English, even when the question is in another language.
3. Use only facts provided by the websearch tool to answer the question. DO NOT use your knowledge about the series.
DO NOT invent anything. 
4. You can modify the query by including more terms or narrowing the topic. If present, use the past messages that regards 
the current question. If you think the query is short, ambiguos or difficult to use in a search, briefly answer the query
and add your response to the query for the next search.
5. If your confidence is 'unsure' or 'low', search again with a different query. DO AT MOST 3 searches.
6. Write the final response in English.
"""

# ============================================= FUNCTIONS =============================================

def format_images(images: List[Dict[str, Any]]) -> List[str]:

    if not images:
        return ""

    if isinstance(images, list):
        return "\n".join([f"image [{i}] url: {image}" for i, image in enumerate(images)])

    if isinstance(images[0], dict):

        images_blocks = []
        for i, image in enumerate(images):
            image_url = image.get("url", "")
            image_caption = image.get("description", "")
            images_blocks.append(
                f"image [{i}] url: {image_url}, description: {image_caption}\n"
            )

        return "\n".join(images_blocks)


def format_results(results: List[Dict[str, Any]]) -> List[str]:
    
    blocks = []
    for i, result in enumerate(sorted(results, key=lambda x: -x["score"])):
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")
        images = format_images(result.get("images", []))

        blocks.append(
            f"[{i}] title: {title}\n"
            f"URL: {url}\n"
            f"images metadata: {images}\n"
            f"content: {content}\n"
        )

    return "\n\n".join(blocks)


def create_web_tool(resources: Resources):

    @tool("websearch", description="Search the web for information regarding the user query")
    async def websearch(query: str, runtime: ToolRuntime) -> List[str]:

        response = await resources.tavily.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_images=True,
        )

        return format_results(response["results"])
            
    return websearch

# ============================================= AGENT =============================================

def create_web_agent(resources: Resources):
    
    main_model = init_chat_model(
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
        model=main_model,
        tools=[create_web_tool(resources)],
        middleware=[
            ToolCallLimitMiddleware(tool_name="websearch", run_limit=3, exit_behavior="continue"),
            ModelCallLimitMiddleware(run_limit=6),
            ModelFallbackMiddleware(first_model=fallback_model),
            ModelRetryMiddleware(max_retries=2)],
        system_prompt=SYSTEM_PROMPT,
        response_format=LoreAnswer,
        name="web_agent"
    )
        
        
# ============================================= TESTING =============================================

async def main():

    from core.resources import open_resources
    from langchain.messages import HumanMessage
    from pprint import pprint

    async with open_resources() as resources:

        agent = create_web_agent(resources)

        question = "Tell me about the butcher's war?"
        messages = [
            HumanMessage(content=question)
        ]

        response = await agent.ainvoke({"messages": messages})

        pprint(response["messages"])
        pprint(response["structured_response"])


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    asyncio.run(main())
