"""
handlers.py
────────────
Four streaming handler coroutines dispatched by the query router.

Each handler is a plain Python generator that yields string tokens —
compatible with FastAPI's StreamingResponse.

Handlers:
    handle_general_llm(query)         → LLM only, no context
    handle_rag(query)                 → Vector search → RAG prompt → LLM
    handle_tool(query, tool_name)     → Execute tool → format result → LLM
    handle_rag_and_tool(query, tool_name) → RAG + tool → merged prompt → LLM
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from app.utils.lc_llm import get_llm
from app.prompts.rag_prompt import RAG_PROMPT
from app.prompts.general_prompt import GENERAL_PROMPT
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.rag.retriever import Retriever
from app.router.tool_registry import get_tool

_retriever = Retriever(top_k=5)


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_rag_context(question: str) -> str:
    """Retrieves and formats document chunks from the vector store."""
    chunks = _retriever.retrieve(question)
    print(f"  [Handler-RAG] Retrieved {len(chunks)} chunk(s).")
    return "\n\n".join(chunks)


def _run_tool(tool_name: str, query: str) -> str:
    """Executes a registered tool and returns its string result."""
    print(f"  [Handler-Tool] Executing tool: '{tool_name}'")
    try:
        fn = get_tool(tool_name)
        result = fn(query)
        print(f"  [Handler-Tool] Tool returned {len(str(result))} chars.")
        return str(result)
    except KeyError:
        print(f"  [Handler-Tool] Tool '{tool_name}' not found — returning empty.")
        return ""
    except Exception as err:
        print(f"  [Handler-Tool] Tool '{tool_name}' error: {err}")
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Handler 1: General LLM (no context)
# ──────────────────────────────────────────────────────────────────────────────

def handle_general_llm(query: str):
    """
    Handles greetings, small talk, and general questions.
    Uses GENERAL_PROMPT — no vector retrieval.
    """
    print("  [Handler] Mode: general_llm")
    try:
        llm = get_llm()
        chain = GENERAL_PROMPT | llm | StrOutputParser()
        for token in chain.stream({"question": query}):
            yield token
    except Exception as err:
        print(f"  [Handler Error - GeneralLLM] {err}")
        yield "Hello! I am Trase, your AI travel assistant. How can I help you with bus routes, schedules, or fares today?"


# ──────────────────────────────────────────────────────────────────────────────
# Handler 2: RAG (vector search + LLM)
# ──────────────────────────────────────────────────────────────────────────────

def handle_rag(query: str):
    """
    Full RAG pipeline: retrieve → RAG prompt → LLM.
    Identical to the existing /chatbot/stream behaviour.
    """
    print("  [Handler] Mode: rag")
    try:
        llm = get_llm()
        chain = (
            {
                "context": RunnableLambda(_get_rag_context),
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )
        for token in chain.stream(query):
            yield token
    except Exception as err:
        print(f"  [Handler Error - RAG] {err}")
        # Try direct database search fallback if the LangChain pipeline hit an issue
        try:
            raw_context = _get_rag_context(query)
            if raw_context:
                yield "Here are the details from our database:\n\n" + raw_context
            else:
                yield "I'm sorry, but those details are not available in our travel database at the moment. Please let me know if there's anything else I can help you find!"
        except Exception:
            yield "I'm experiencing high server demand at the moment. Please try asking your question again in a few seconds."


# ──────────────────────────────────────────────────────────────────────────────
# Handler 3: Tool only
# ──────────────────────────────────────────────────────────────────────────────

_TOOL_ONLY_TEMPLATE = """\
You are Trase, a bus travel assistant.
A tool was called to answer the user's request and returned the following result:

Tool Result:
{tool_result}

User Question:
{question}

Instructions:
- Summarise the tool result in a clear, friendly way for the user.
- If the tool result is empty or unhelpful, say you were unable to retrieve \
  that information right now.

Answer:"""

_TOOL_ONLY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", _TOOL_ONLY_TEMPLATE),
])


def handle_tool(query: str, tool_name: str):
    """
    Executes the named tool, then uses the LLM to format the result
    into a friendly response for the user.
    """
    print(f"  [Handler] Mode: tool ('{tool_name}')")
    try:
        tool_result = _run_tool(tool_name, query)
        llm = get_llm()
        chain = _TOOL_ONLY_PROMPT | llm | StrOutputParser()
        for token in chain.stream({"question": query, "tool_result": tool_result}):
            yield token
    except Exception as err:
        print(f"  [Handler Error - Tool] {err}")
        yield "I encountered an issue executing that tool. Please try asking directly about specific bus routes or schedules."


# ──────────────────────────────────────────────────────────────────────────────
# Handler 4: RAG + Tool (merged context)
# ──────────────────────────────────────────────────────────────────────────────

_RAG_AND_TOOL_TEMPLATE = """\
Context from Travel Database:
{context}

Tool Result:
{tool_result}

User Question:
{question}

Instructions:
- Answer using BOTH the database context AND the tool result.
- Prefer the tool result for computed/exact values (e.g. fares).
- Prefer the database context for descriptive information (e.g. facilities).
- Never mention internal IDs, route codes, or database _ids.
- If details are unavailable, politely say so.

Answer:"""

_RAG_AND_TOOL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", _RAG_AND_TOOL_TEMPLATE),
])


def handle_rag_and_tool(query: str, tool_name: str):
    """
    Retrieves vector context AND runs a tool, then merges both into the
    LLM prompt for the richest possible answer.
    """
    print(f"  [Handler] Mode: rag_and_tool ('{tool_name}')")
    try:
        context = _get_rag_context(query)
        tool_result = _run_tool(tool_name, query)
        llm = get_llm()
        chain = _RAG_AND_TOOL_PROMPT | llm | StrOutputParser()
        for token in chain.stream({
            "question": query,
            "context": context,
            "tool_result": tool_result,
        }):
            yield token
    except Exception as err:
        print(f"  [Handler Error - RAG+Tool] {err}")
        yield from handle_rag(query)
