from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import (ModelRetryMiddleware, 
                                        ModelFallbackMiddleware,
                                        ModelCallLimitMiddleware)

from agents.contracts import LoreAnswer
from core.resources import Resources

import httpx
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)



# ================================ MODELS ================================

MODEL_NAME = "nvidia/nemotron-3.5-lightning:free"
FALLBACK_NAME = "xiaomi/mimo-v2.6-flash"

# ================================ API UTILS ================================

class MetronError(Exception):
    """Errore che non significa "non trovato": token, limiti, server."""

async def metron_get(client: httpx.AsyncClient, endpoint: str, params: Dict[str, Any] = None) -> str:
    
    try:
        response = await client.get(
            endpoint,
            params=params
        )

    except httpx.HTTPStatusError as exc:
        raise MetronError(f"HTTP {exc.response.status_code} for {endpoint}: {exc}")
    
    if response.status_code == 404:
        return None
    
    if response.status_code in (401, 403):
        raise MetronError(f"Metron rejected the token for {endpoint}: {response.text}")
    
    if response.status_code == 429:
        raise MemoryError(f"Rate limit exceeded for {endpoint}: {response.text}")

    if response.status_code >= 400:
        raise MetronError(f"Metron returned HTTP {response.status_code}.")

    return response.json()


async def search_and_detail(client: httpx.AsyncClient, search_param: str, endpoint: str) -> List[Dict[str, Any]]:

    page = await metron_get(client, f"{endpoint}/", {"name": search_param})

    if page is None:
        return []

    page_results = page.get("results", [])
    
    results = []
    for result in page_results:
        if search_id := result.get("id", None):
            detail = await metron_get(client, f"{endpoint}/{search_id}/")
            results.append(detail)

    return results

# ================================ UTILS ================================

def _names(items: Optional[List[Dict[str, Any]]]) -> str:
    return ", ".join(i.get("name", "") for i in items or [] if i.get("name"))
 
 
def _lines(*pairs: tuple[str, Any]) -> str:
    """Una riga "Etichetta: valore" per ogni valore non vuoto."""
    return "\n".join(f"{label}: {value}" for label, value in pairs if value not in (None, "", []))


def format_character_detail(c: Dict[str, Any]) -> str:
    return _lines(
        ("Name", c.get("name")),
        ("Aliases", ", ".join(c.get("alias") or [])),
        ("Description", c.get("desc")),
        ("Creators", _names(c.get("creators"))),
        ("Teams", _names(c.get("teams"))),
        ("Resource URL", c.get("resource_url")),
    )
 

def format_creator_detail(c: Dict[str, Any]) -> str:
    return _lines(
        ("Name", c.get("name")),
        ("Birth date", c.get("birth")),
        ("Death date", c.get("death")),
        ("Description", c.get("desc")),
        ("Resource URL", c.get("resource_url")),
    )
 
 
def format_series_detail(s: Dict[str, Any]) -> str:
    return _lines(
        ("Name", s.get("name")),
        ("Volume", s.get("volume")),
        ("Type", (s.get("series_type") or {}).get("name")),
        ("Status", s.get("status")),
        ("Publisher", (s.get("publisher") or {}).get("name")),
        ("Year started", s.get("year_began")),
        ("Year ended", s.get("year_end") or "Ongoing"),
        ("Number of issues", s.get("issue_count")),
        ("Genres", _names(s.get("genres"))),
        ("Description", s.get("desc")),
        ("Resource URL", s.get("resource_url")),
    )
 
 
def format_arc_detail(a: Dict[str, Any]) -> str:
    return _lines(
        ("Name", a.get("name")),
        ("Description", a.get("desc")),
        ("Resource URL", a.get("resource_url")),
    )


def format_issue_detail(issue: Dict[str, Any]) -> str:

    series = issue.get("series") or {}
    credits = "; ".join(
        f"{(c.get('creator') or '')} ({', '.join(r.get('name', '') for r in c.get('role') or [])})"
        for c in issue.get("credits") or []
    )
    return _lines(
        ("Series", series.get("name")),
        ("Number", issue.get("number")),
        ("Story titles", ", ".join(issue.get("name") or [])),
        ("Store date", issue.get("store_date")),
        ("Cover date", issue.get("cover_date")),
        ("Publisher", (issue.get("publisher") or {}).get("name")),
        ("Story arcs", _names(issue.get("arcs"))),
        ("Credits", credits),
        ("Characters", _names(issue.get("characters"))),
        ("Description", issue.get("desc")),
        ("Variant covers", len(issue.get("variants") or []) or None),
        ("Resource URL", issue.get("resource_url")),
    )


def format_issue_list(issues: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"#{i.get('number', '?')} | store date {i.get('store_date') or '?'} | cover date {i.get('cover_date') or '?'}"
        f" | {i.get('issue', '')}"
        for i in issues
    )


# ================================ TOOLS ================================

def create_agent_tools(resources: Resources):

    metron_client = resources.metron


    @tool("character", description="Get all character detail from character name")
    async def get_character_detail(character_name: str, runtime:ToolRuntime) -> str:
        
        try:
            character_details = await search_and_detail(
                metron_client,
                character_name,
                "/character"
            )
        except MetronError as error:
            return f"Error calling Metron API: {error}"
        
        if not character_details:
            return f"NO RESULTS: Metron does not contain any character named {character_name}"

        return "\n\n".join([format_character_detail(c) for c in character_details])



    @tool("arc", description="Search for narrative arc id and name from Metron")
    async def get_narrative_arc(arc_name: str, runtime: ToolRuntime) -> str:

        try:
            arc_details = await search_and_detail(
                metron_client,
                arc_name,
                "/arc"
            )
        except MetronError as error:
            return f"Error calling Metron API: {error}"

        if not arc_details:
            return f"NO RESULTS: Metron does not contain any arc named {arc_name}"

        return "\n\n".join([format_arc_detail(a) for a in arc_details])


    @tool("creator", description="Get all creator detail from creator name")
    async def get_creator_detail(creator_name: str, runtime: ToolRuntime) -> str:

        try:
            creator_detail = await search_and_detail(
                metron_client,
                creator_name,
                "/creator"
            )
        except MetronError as error:
            return f"Error calling Metron API: {error}"
        
        if not creator_detail:
            return f"NO RESULTS: Metron does not contain any creator named {creator_name}"

        return "\n\n".join([format_creator_detail(c) for c in creator_detail])


    @tool("issues", description="Get all issue detail from issue name")
    async def get_issue_detail(issue_name: str, runtime: ToolRuntime) -> str:

        try:
            issue_details = await search_and_detail(
                metron_client,
                issue_name,
                "/issue"
            )
        except MetronError as error:
            return f"Error calling Metron API: {error}"
        
        if not issue_details:
            return f"NO RESULTS: Metron does not contain any issue named {issue_name}"

        return "\n\n".join([format_issue_detail(i) for i in issue_details])


    @tool("series", description="Get all series details from a series name.")
    async def get_series_detail(series_name: str, runtime: ToolRuntime) -> str:

        try:
            series_details = await search_and_detail(
                metron_client,
                series_name,
                "/series"
            )
        except MetronError as error:
            return f"Error calling Metron API: {error}"
        
        if not series_details:
            return f"NO RESULTS: Metron does not contain any series named {series_name}"

        return "\n\n".join([format_series_detail(s) for s in series_details])


    @tool("series_issues", description="List every issue of a series with its number, store date and cover date.")
    async def series_issues(series_name: str, runtime: ToolRuntime) -> str:
        """L"""
        try:
            issues = await metron_get(metron_client, "issue/", {"series_name": series_name})
        except MetronError as exc:
            return f"ERROR: {exc}"

        if not issues:
            return f"NO_RESULTS: no issues found for series {series_name!r}."
    
        return f"{len(issues)} issues:\n {format_issue_list(issues)}"

    return [get_character_detail, get_creator_detail, get_issue_detail, get_series_detail, series_issues]


# ================================ PROMPT ===============================

SYSTEM_PROMPT = """You are a helpful assistant. Your answer questions about comic series  
SIKTC: Something is Killing the Children (Boom! Studios) and its spin-offs (House of Slaughter, Fall of the House of Slaughter, etc.) 
only by accessing the Metron comic database via tools.

Tools:
- character, creator, series, issue: look up one entity by name and return all details.
- series_issues: every issue of a series with its dates.

For each search, provide:
- 'answer': the answer to the question posed by the user using the detailed information retrieved from the tools. DO NOT include any reasoning.
- 'retrieved': the Resource URLs you actually used to answer the question.
- 'confidence': how confident you are about the answer. Reply with 'high' if the retrieved information provide a clear answer, 
'unsure' if they provide a partial answer, and 'low' if they are off-topics, they provide a very limited answer or no answer.
The 'answer' MUST BE 2-4 sentences, about 250 words long.

Follow these rules:
1. Always call the tools before providing an answer. Be aware that all tools return a list of results.
2. Search in English and use full English names in tool arguments (e.g. "Erica Slaughter", "Something Is Killing the Children").
Use the earlier messages to resolve references such as "she" or "that arc" if present.
3. Use only facts provided by the tools to answer the question.  Do NOT add your own knowledge of the series. DO NOT invent anything.
4. If a tool returns NO_RESULTS, try once more with a shorter or alternative name; then stop and say
   that Metron does not cover it. If a tool returns ERROR, do not retry: report it.
5. Answer in English.
"""


# ================================ AGENTS ================================

def create_comic_agent(resources: Resources):
    
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
        tools=create_agent_tools(resources),
        middleware=[
            ModelCallLimitMiddleware(run_limit=10),
            ModelFallbackMiddleware(first_model=fallback_model),
            ModelRetryMiddleware(max_retries=2)],
        system_prompt=SYSTEM_PROMPT,
        response_format=LoreAnswer,
        name="comics_agent"
    )


# ==================================================== TEST ====================================================

async def main() -> None:
    from langchain.messages import HumanMessage
    from core.resources import open_resources
 
    async with open_resources() as resources:
        agent = create_comic_agent(resources)
        response = await agent.ainvoke(
            {"messages": [HumanMessage(content="Who created the character Erica Slaughter?")]})
        print(response.get("structured_response"))
 
 
if __name__ == "__main__":

    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        force=True)

    logging.getLogger("httpx").setLevel(logging.WARNING)

    asyncio.run(main())
