"""API FastAPI per il coordinator SIKTC.

Endpoint:
- GET  /api/health
- POST /api/chat/stream   NDJSON: una riga JSON per evento (usato dalla pagina)
- POST /api/chat          stessa risposta finale in un unico JSON (comodo per curl e test)

Eventi dello stream:
  {"type": "start",  "thread_id": "..."}
  {"type": "agent",  "agent": "lore|comic|web|coordinator", "state": "active|done", "query": "..."}
  {"type": "result", "data": { ...payload della pagina... }}
  {"type": "error",  "status": 502, "message": "..."}
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import unquote, urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from agents.coordinator import build_coordinator_agent
from core.resources import open_resources


logger = logging.getLogger(__name__)


# ============================================= CONFIG =============================================

STATIC_DIR = Path(os.getenv("STATIC_DIR", Path(__file__).resolve().parents[2] / "static"))           # la pagina: static/index.html
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "240"))  # secondi per una risposta completa
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]

# Tool del coordinator -> id dell'agente mostrato nella pagina
AGENT_OF_TOOL = {
    "call_lore_subagent": "lore",
    "call_comic_subagent": "comic",
    "call_web_subagent": "web",
}

CITE_RE = re.compile(r"(\s*)\[(\d+(?:\s*,\s*\d+)*)\]")


# ============================================= SCHEMAS =============================================

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000, description="Domanda dell'utente")
    thread_id: Optional[str] = Field(default=None, max_length=64, description="Conversazione da continuare; vuoto = nuova")
    spoiler_limit: Optional[str] = Field(default=None, pattern=r"^(all|\d{1,4})$",
                                         description="Ultimo albo letto, oppure 'all'")


class RunFailed(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# ============================================= LETTURA DEI RISULTATI =============================================

def parse_subagent_result(content: str) -> Optional[Dict[str, Any]]:
    """Legge il testo prodotto da format_subagent_answer nel coordinator.

    Restituisce None per i messaggi che non vengono da un subagente
    (per esempio il rifiuto di ToolCallLimitMiddleware su una seconda chiamata).
    """
    text = content.strip()
    if text.lower().startswith("error"):
        return {"status": "error", "confidence": None, "urls": []}
    if not text.startswith("SUBAGENT:"):
        return None

    confidence = re.search(r"^CONFIDENCE:\s*(\w+)", text, re.M)
    block = re.search(r"^SOURCE URLS:\n(.*?)(?=^IMAGE URLS:|\Z)", text, re.M | re.S)
    urls = []
    if block:
        for line in block.group(1).splitlines():
            line = line.strip()
            if line.startswith("- ") and line != "- (none)":
                urls.append(line[2:].strip())

    return {"status": "ok", "confidence": confidence.group(1).lower() if confidence else None, "urls": urls}


def describe_source(url: str, found_by: List[str]) -> Dict[str, Any]:
    """Titolo, dominio e tipo di una fonte ricavati dal suo URL."""
    parts = urlsplit(url)
    host = (parts.hostname or "").removeprefix("www.")
    segment = unquote(next((s for s in reversed(parts.path.split("/")) if s), ""))

    if "fandom.com" in host or "wikipedia.org" in host:
        kind, title = "Wiki", segment.replace("_", " ")
    elif "metron.cloud" in host:
        kind, title = "Metron", segment.replace("-", " ").title()
    else:
        kind = "Web"
        title = re.sub(r"\.(html?|php|aspx?)$", "", segment).replace("-", " ").replace("_", " ").strip()
        title = title[:1].upper() + title[1:]

    title = title or host or url
    if len(title) > 90:
        title = title[:88].rstrip() + "…"

    return {"title": title, "url": url, "domain": host, "kind": kind, "found_by": found_by}


def build_payload(answer: Any, trace: List[Dict[str, Any]], elapsed_ms: int, thread_id: str) -> Dict[str, Any]:
    """Converte AnnotatedAnswer nel formato della pagina."""

    # Quale agente ha trovato ogni URL
    found_by: Dict[str, List[str]] = {}
    for step in trace:
        for url in step.pop("urls", []):
            found_by.setdefault(url, []).append(step["agent"])

    # Fonti ordinate e numerate da 1, senza doppioni; remap dei numeri nel testo
    remap: Dict[int, int] = {}
    position: Dict[str, int] = {}
    sources: List[Dict[str, Any]] = []
    for source in sorted(answer.sources or [], key=lambda s: s.id):
        url = (source.url or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        if url not in position:
            sources.append(describe_source(url, found_by.get(url, [])))
            position[url] = len(sources)
        remap[source.id] = position[url]

    def renumber(match: re.Match) -> str:
        numbers = {remap.get(int(n)) for n in re.split(r"\s*,\s*", match.group(2))}
        cites = "".join(f"[{n}]" for n in sorted(n for n in numbers if n))
        return match.group(1) + cites if cites else ""   # citazione a una fonte inesistente: sparisce con il suo spazio

    text = CITE_RE.sub(renumber, answer.answer or "")
    paragraphs = [{"text": p.strip()} for p in re.split(r"\n\s*\n", text) if p.strip()]

    consulted = [s for s in trace if s["status"] == "ok"]
    trace.append({"agent": "coordinator", "status": "ok", "ms": elapsed_ms, "sources": len(sources)})

    return {
        "thread_id": thread_id,
        "elapsed_ms": elapsed_ms,
        "paragraphs": paragraphs,
        "sources": sources,
        "images": [u for u in (answer.images or []) if u.startswith(("http://", "https://"))][:2],
        "trace": trace,
        "agreement": {"agree": sum(s["confidence"] == "high" for s in consulted), "total": len(consulted)},
        "followups": list(answer.next_question or [])[:3],
    }


def _updates(value: Any) -> List[Dict[str, Any]]:
    """Un aggiornamento di nodo può essere un dict, una lista o None."""
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, dict)]
    return []


# ============================================= ESECUZIONE =============================================

Emit = Callable[[Dict[str, Any]], None]


def build_message(body: ChatRequest) -> str:
    """Lo scudo anti-spoiler arriva al coordinator come nota in coda alla domanda."""
    if body.spoiler_limit and body.spoiler_limit != "all":
        return (f"{body.message}\n\n(Spoiler shield: I have read up to issue #{body.spoiler_limit}. "
                "Do not reveal events from later issues. Answer in the language of my question above.)")
    return body.message


async def run_question(agent: Any, thread_id: str, message: str, emit: Emit) -> Dict[str, Any]:
    """Esegue il coordinator in streaming e segnala quando ogni agente parte e finisce."""

    config = {"configurable": {"thread_id": thread_id}}
    calls: Dict[str, Dict[str, Any]] = {}   # tool_call_id -> agente in corso
    trace: List[Dict[str, Any]] = []
    structured = None
    start = time.perf_counter()

    emit({"type": "agent", "agent": "coordinator", "state": "active"})

    async for chunk in agent.astream({"messages": [HumanMessage(content=message)]},
                                     config=config, stream_mode="updates"):
        for node, value in chunk.items():
            for update in _updates(value):

                if update.get("structured_response") is not None:
                    structured = update["structured_response"]

                # Solo i nodi del modello e dei tool: il riassunto della SummarizationMiddleware
                # ripubblica i messaggi vecchi, che non devono riaccendere gli agenti.
                for msg in update.get("messages") or []:

                    if node == "model" and isinstance(msg, AIMessage):
                        for call in msg.tool_calls:
                            agent_id = AGENT_OF_TOOL.get(call["name"])
                            if agent_id and call["id"] not in calls:
                                query = str((call.get("args") or {}).get("query", ""))
                                calls[call["id"]] = {"agent": agent_id, "query": query, "t0": time.perf_counter()}
                                emit({"type": "agent", "agent": agent_id, "state": "active", "query": query})

                    elif node == "tools" and isinstance(msg, ToolMessage):
                        call = calls.pop(msg.tool_call_id, None)
                        if call is None:
                            continue
                        parsed = parse_subagent_result(str(msg.content))
                        if parsed is not None:
                            trace.append({
                                "agent": call["agent"],
                                "query": call["query"],
                                "status": parsed["status"],
                                "confidence": parsed["confidence"],
                                "urls": parsed["urls"],
                                "ms": round((time.perf_counter() - call["t0"]) * 1000),
                            })
                        emit({"type": "agent", "agent": call["agent"], "state": "done"})

    for call in calls.values():   # chiamate rimaste senza risposta
        emit({"type": "agent", "agent": call["agent"], "state": "done"})

    if structured is None:
        snapshot = await agent.aget_state(config)
        structured = (snapshot.values or {}).get("structured_response")
    if structured is None:
        raise RunFailed(502, "Il modello non ha prodotto una risposta valida. Prova a riformulare la domanda.")

    emit({"type": "agent", "agent": "coordinator", "state": "done"})
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    return build_payload(structured, trace, elapsed_ms, thread_id)


async def start_run(request: Request, body: ChatRequest) -> asyncio.Queue:
    """Avvia la domanda in un task separato e restituisce la coda dei suoi eventi.

    Il task non dipende dalla connessione: se il browser interrompe l'indagine,
    il coordinator finisce comunque il suo giro e la conversazione resta coerente
    (un run cancellato a metà lascerebbe chiamate ai tool senza risposta nel checkpointer).
    """
    state = request.app.state
    thread_id = body.thread_id or uuid.uuid4().hex

    lock = state.thread_locks.setdefault(thread_id, asyncio.Lock())
    if lock.locked():
        raise HTTPException(status_code=409, detail="Gli agenti stanno ancora lavorando alla domanda precedente. Riprova tra poco.")
    await lock.acquire()   # libero: si acquisisce subito, senza cedere il controllo

    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait({"type": "start", "thread_id": thread_id})

    async def worker() -> None:
        try:
            payload = await asyncio.wait_for(
                run_question(state.agent, thread_id, build_message(body), queue.put_nowait),
                timeout=AGENT_TIMEOUT,
            )
            queue.put_nowait({"type": "result", "data": payload})
        except asyncio.TimeoutError:
            logger.warning("Timeout sul thread %s", thread_id)
            queue.put_nowait({"type": "error", "status": 504,
                              "message": "La risposta ha richiesto troppo tempo. Riprova con una domanda più specifica."})
        except RunFailed as exc:
            logger.error("Run fallito sul thread %s: %s", thread_id, exc.message)
            queue.put_nowait({"type": "error", "status": exc.status, "message": exc.message})
        except Exception:
            logger.exception("Errore del coordinator sul thread %s", thread_id)
            queue.put_nowait({"type": "error", "status": 502, "message": "Errore durante la generazione della risposta."})
        finally:
            lock.release()
            queue.put_nowait(None)   # fine dello stream

    task = asyncio.create_task(worker())
    state.tasks.add(task)
    task.add_done_callback(state.tasks.discard)
    return queue


# ============================================= APP =============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Apre le risorse e costruisce il coordinator una sola volta, all'avvio."""
    async with open_resources() as resources:
        app.state.agent = build_coordinator_agent(resources)
        app.state.thread_locks = {}
        app.state.tasks = set()
        logger.info("Coordinator pronto")
        yield
        if app.state.tasks:   # lascia finire le domande in corso prima di chiudere le risorse
            await asyncio.wait(app.state.tasks, timeout=AGENT_TIMEOUT)
    logger.info("Risorse chiuse")


app = FastAPI(title="Something Is Killing the Chatbot", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    queue = await start_run(request, body)

    async def events():
        while (event := await queue.get()) is not None:
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},   # niente buffering dietro nginx
    )


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request) -> dict:
    queue = await start_run(request, body)
    while (event := await queue.get()) is not None:
        if event["type"] == "result":
            return event["data"]
        if event["type"] == "error":
            raise HTTPException(status_code=event["status"], detail=event["message"])
    raise HTTPException(status_code=502, detail="Nessuna risposta dagli agenti.")


# ============================================= PAGINA =============================================

# Montato per ultimo: le route /api/* hanno la precedenza, "/" serve static/index.html.
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:
    logger.warning("Cartella %s assente: la pagina non viene servita, solo le API.", STATIC_DIR)
