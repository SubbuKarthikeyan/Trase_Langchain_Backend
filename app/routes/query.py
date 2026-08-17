"""
query.py
─────────
New FastAPI router: POST /query/stream

This endpoint is the intelligent entry point for all user queries.
It uses the intent classifier to route each query to the correct handler:

    general_llm  → handle_general_llm()
    rag          → handle_rag()
    tool         → handle_tool()
    rag_and_tool → handle_rag_and_tool()

Accepts a JSON body (more robust than query parameters):
    POST /query/stream
    { "message": "Are there any AC buses from Chennai to Bangalore?" }

The original /chatbot/stream endpoint is preserved for backward compatibility.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.router.intent_classifier import classify_intent
from app.router.handlers import (
    handle_general_llm,
    handle_rag,
    handle_tool,
    handle_rag_and_tool,
)

router = APIRouter(prefix="/query", tags=["Query Router"])


# ──────────────────────────────────────────────────────────────────────────────
# Request schema
# ──────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    message: str


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def _dispatch(message: str):
    """
    Classifies the query intent and yields streaming tokens from the
    appropriate handler. This is the core routing function.
    """
    print("=" * 55)
    print(f"[QueryRouter] Received: {message!r}")

    classification = classify_intent(message)
    intent = classification.get("intent", "rag")
    tool_name = classification.get("tool_name")

    print(f"[QueryRouter] Dispatching -> intent='{intent}', tool='{tool_name}'")

    if intent == "general_llm":
        yield from handle_general_llm(message)

    elif intent == "tool":
        if tool_name:
            yield from handle_tool(message, tool_name)
        else:
            # tool intent but no tool name — safe fallback to RAG
            print("  [QueryRouter] tool_name is None, falling back to rag.")
            yield from handle_rag(message)

    elif intent == "rag_and_tool":
        if tool_name:
            yield from handle_rag_and_tool(message, tool_name)
        else:
            print("  [QueryRouter] tool_name is None for rag_and_tool, falling back to rag.")
            yield from handle_rag(message)

    else:
        # Default: "rag" (also covers unknown intents)
        yield from handle_rag(message)

    print("[QueryRouter] Response complete.")
    print("=" * 55)


# ──────────────────────────────────────────────────────────────────────────────
# Route
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/stream")
def query_stream(body: QueryRequest):
    """
    Smart streaming endpoint — classifies query intent and routes to the
    correct pipeline: GeneralLLM, RAG, ToolCall, or RAG+ToolCall.

    Request body:
        { "message": "Your question here" }
    """
    return StreamingResponse(
        _dispatch(body.message),
        media_type="text/plain",
    )


@router.get("/health")
def query_router_health():
    """Returns the list of currently registered tools."""
    from app.router.tool_registry import list_tools
    return {
        "status": "ok",
        "registered_tools": list_tools(),
    }
