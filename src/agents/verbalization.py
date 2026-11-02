import logging
import time

from langchain.agents.middleware import after_model, wrap_tool_call
from langchain_core.messages import AIMessage


trace_log = logging.getLogger("trace")


def make_narrator(agent_name: str) -> list:
    """Middleware di debug: racconta decisioni del modello e durata dei tool."""

    @after_model(name=f"{agent_name}_narrator")
    def narrate(state, runtime):
        msg = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        if msg is None:
            return None
        if thinking := msg.additional_kwargs.get("reasoning_content"):
            trace_log.info("[%s] pensiero: %s", agent_name, thinking[:500])
        if msg.content:
            trace_log.info("[%s] testo: %s", agent_name, str(msg.content)[:300])
        for call in msg.tool_calls:
            trace_log.info("[%s] chiama %s(%s)", agent_name, call["name"], call["args"])
        trace_log.info("[%s] uso token: %s", agent_name, msg.usage_metadata)
        return None

    @wrap_tool_call(name=f"{agent_name}_tool_timer")
    async def time_tools(request, handler):
        start = time.perf_counter()
        result = await handler(request)
        trace_log.info("[%s] %s finito in %.1fs", agent_name, request.tool_call["name"],
                       time.perf_counter() - start)
        return result

    return [narrate, time_tools]
