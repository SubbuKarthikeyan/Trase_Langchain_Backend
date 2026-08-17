"""
intent_classifier.py
─────────────────────
LLM-based query intent classifier for the Trase router.

Classifies each user query into one of four intents:
    - "general_llm"  : Greetings, small talk, general travel advice
    - "rag"          : Questions about specific bus routes, schedules, fares
    - "tool"         : Requests that need a registered tool to answer
    - "rag_and_tool" : Needs both vector search context AND a tool call

The classifier makes a single fast LLM call with a structured JSON prompt.
Falls back to "rag" on any error (safest default for a travel chatbot).

Design:
    - Tool descriptions are injected dynamically from tool_registry, so adding
      a new tool requires zero changes here.
    - Falls back to a keyword-based heuristic if the LLM call fails, ensuring
      zero downtime even during LLM outages.
"""

import json
import re

from app.utils.lc_llm import get_llm
from app.router.tool_registry import get_tool_descriptions, has_tools


# ──────────────────────────────────────────────────────────────────────────────
# Classifier prompt builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_classifier_prompt(query: str) -> str:
    """
    Builds a tightly-scoped classification prompt.
    Injects available tool descriptions dynamically.
    """
    tool_section = ""
    if has_tools():
        descriptions = get_tool_descriptions()
        tool_lines = "\n".join(
            f'    - "{name}": {desc}' for name, desc in descriptions.items()
        )
        tool_section = f"""
Available tools:
{tool_lines}

Use "tool" if the query specifically needs one of these tools.
Use "rag_and_tool" if the query needs BOTH database context AND a tool.
"""
    else:
        tool_section = (
            "\nNo tools are currently registered. "
            "Never classify as 'tool' or 'rag_and_tool'.\n"
        )

    return f"""You are a query intent classifier for a bus travel chatbot called Trase.

Classify the user query into EXACTLY ONE of these intents:
  - "general_llm"  : Greetings, small talk, thank you, compliments, general travel tips not needing bus data
  - "rag"          : Questions about specific bus routes, schedules, departure/arrival times, fares, seats, facilities, stops
  - "tool"         : Queries that require calling a specific tool to compute or fetch structured data
  - "rag_and_tool" : Queries that need BOTH retrieved bus data AND a tool call
{tool_section}
Respond with ONLY valid JSON in this exact format (no explanation, no markdown):
{{"intent": "<one of: general_llm, rag, tool, rag_and_tool>", "tool_name": <null or "tool_name_string">}}

User query: "{query}"

JSON:"""


# ──────────────────────────────────────────────────────────────────────────────
# Keyword fallback (used when LLM classification fails)
# ──────────────────────────────────────────────────────────────────────────────

_GENERAL_KEYWORDS = {
    "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye",
    "good morning", "good evening", "good afternoon", "how are you",
    "what can you do", "help", "who are you",
}

_RAG_KEYWORDS = {
    "bus", "route", "schedule", "fare", "ticket", "seat", "stop",
    "departure", "arrival", "timing", "time", "from", "to", "travel",
    "book", "available", "price", "cost", "ac", "sleeper", "ordinary",
    "deluxe", "facilities", "wifi", "charging",
}


def _keyword_fallback(query: str) -> dict:
    """
    Simple keyword-based classifier used when LLM is unavailable.
    Defaults to 'rag' if neither general nor tool keywords match.
    """
    q = query.lower()
    if any(kw in q for kw in _GENERAL_KEYWORDS) and not any(kw in q for kw in _RAG_KEYWORDS):
        return {"intent": "general_llm", "tool_name": None}
    return {"intent": "rag", "tool_name": None}


# ──────────────────────────────────────────────────────────────────────────────
# Public classifier
# ──────────────────────────────────────────────────────────────────────────────

def classify_intent(query: str) -> dict:
    """
    Classifies the user query and returns a dict:
        {
            "intent":    "general_llm" | "rag" | "tool" | "rag_and_tool",
            "tool_name": None | "tool_name_string"
        }

    Guarantees never to raise — always returns a valid classification.
    """
    print(f"\n[Router] Classifying intent for: '{query}'")

    try:
        llm = get_llm()
        prompt = _build_classifier_prompt(query)

        # Use the underlying LLM directly (non-streaming, single call)
        response = llm.invoke(prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)

        # Extract JSON from response (handle models that add extra text)
        json_match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in LLM response: {raw_text!r}")

        result = json.loads(json_match.group())

        # Validate intent value
        valid_intents = {"general_llm", "rag", "tool", "rag_and_tool"}
        if result.get("intent") not in valid_intents:
            raise ValueError(f"Invalid intent: {result.get('intent')!r}")

        # Guard: if no tools registered, downgrade tool intents to rag
        if not has_tools() and result["intent"] in ("tool", "rag_and_tool"):
            print(f"  [Router] Downgrading '{result['intent']}' -> 'rag' (no tools registered)")
            result = {"intent": "rag", "tool_name": None}

        print(f"  [Router] Intent: {result['intent']} | Tool: {result.get('tool_name')}")
        return result

    except Exception as err:
        print(f"  [Router] Classification failed ({err}), using keyword fallback.")
        fallback = _keyword_fallback(query)
        print(f"  [Router] Fallback intent: {fallback['intent']}")
        return fallback
