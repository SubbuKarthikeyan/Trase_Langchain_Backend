"""
handlers.py
────────────
Four streaming handler coroutines dispatched by the query router.

Each handler is a plain Python generator that yields string tokens —
compatible with FastAPI's StreamingResponse. Integrated with session_store
for multi-turn memory retention and AgentMail email tool dispatch.
"""

import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.utils.lc_llm import get_llm
from app.prompts.rag_prompt import RAG_PROMPT
from app.prompts.general_prompt import GENERAL_PROMPT
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.rag.retriever import Retriever
from app.router.tool_registry import get_tool
from app.utils.session_memory import session_store

_retriever = Retriever(top_k=5)


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_rag_context(question: str) -> str:
    """Retrieves and formats document chunks from vector store / hybrid RRF engine."""
    chunks = _retriever.retrieve(question)
    mode = "Hybrid RRF" if _retriever.hybrid else "Waterfall"
    print(f"  [Handler-RAG] Retrieved {len(chunks)} chunk(s) via {mode} search.")
    return "\n\n".join(chunks)


def _run_tool(tool_name: str, query: str) -> str:
    """Executes a registered tool and returns its string result with terminal logging."""
    print(f"\n  [Handler-Tool] Executing tool: '{tool_name}' for query: '{query}'")
    try:
        fn = get_tool(tool_name)
        
        # Special handling for send_email tool
        if tool_name == "send_email":
            # Extract email address
            email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", query)
            target_email = email_match.group(0) if email_match else (session_store.user_email or "user@example.com")
            session_store.user_email = target_email

            # Gather details from stored selection or conversation history
            bus_details = session_store.get_selected_bus()
            if not bus_details:
                bus_details = session_store.get_history_text(max_messages=6)

            print(f"  [Handler-Tool EMAIL] Target Email: {target_email}")
            result = fn(to_email=target_email, bus_details=bus_details)
            print(f"  [Handler-Tool EMAIL RESULT] {result}")
            return str(result)
        else:
            result = fn(query)
            print(f"  [Handler-Tool] Tool returned {len(str(result))} chars.")
            return str(result)

    except KeyError:
        print(f"  [Handler-Tool ERROR] Tool '{tool_name}' not registered.")
        return f"ERROR: Tool '{tool_name}' is not registered."
    except Exception as err:
        print(f"  [Handler-Tool ERROR] Execution failed for '{tool_name}': {err}")
        return f"ERROR: Failed to execute tool '{tool_name}': {str(err)}"


# ──────────────────────────────────────────────────────────────────────────────
# Handler 1: General LLM (no vector search)
# ──────────────────────────────────────────────────────────────────────────────

def handle_general_llm(query: str):
    """Handles greetings, small talk, and general queries with session memory."""
    print("  [Handler] Mode: general_llm")
    session_store.add_user_message(query)
    chat_history = session_store.get_history_text()

    full_response = ""
    try:
        llm = get_llm()
        chain = GENERAL_PROMPT | llm | StrOutputParser()
        for token in chain.stream({"question": query, "chat_history": chat_history}):
            full_response += token
            yield token
    except Exception as err:
        print(f"  [Handler Error - GeneralLLM] {err}")
        fallback = "Hello! I am Trase, your AI travel assistant. How can I help you with bus routes, schedules, or fares today?"
        full_response = fallback
        yield fallback
    finally:
        session_store.add_assistant_message(full_response)


# ──────────────────────────────────────────────────────────────────────────────
# Handler 2: RAG (vector search + LLM)
# ──────────────────────────────────────────────────────────────────────────────

def handle_rag(query: str):
    """Full RAG pipeline with session memory."""
    print("  [Handler] Mode: rag")
    session_store.add_user_message(query)
    chat_history = session_store.get_history_text()

    full_response = ""
    try:
        context = _get_rag_context(query)
        # Store context if user selected an option or asked for bus details
        if "option" in query.lower() or any(w in query.lower() for w in ["choose", "select", "want", "book"]):
            session_store.set_selected_bus(context[:1500])

        llm = get_llm()
        chain = RAG_PROMPT | llm | StrOutputParser()
        for token in chain.stream({
            "context": context,
            "question": query,
            "chat_history": chat_history
        }):
            full_response += token
            yield token
    except Exception as err:
        print(f"  [Handler Error - RAG] {err}")
        try:
            raw_context = _get_rag_context(query)
            if raw_context:
                fallback = "Here are the details from our database:\n\n" + raw_context
            else:
                fallback = "I'm sorry, but those details are not available in our travel database at the moment."
        except Exception:
            fallback = "I'm experiencing high server demand right now. Please try again in a few moments."
        full_response = fallback
        yield fallback
    finally:
        session_store.add_assistant_message(full_response)


# ──────────────────────────────────────────────────────────────────────────────
# Handler 3: Tool only
# ──────────────────────────────────────────────────────────────────────────────

_TOOL_ONLY_TEMPLATE = """\
Recent Chat History:
{chat_history}

Tool Execution Result:
{tool_result}

User Request:
{question}

Instructions:
- Summarise the tool result in a clear, friendly, and concise way.
- Confirm clearly if an email was sent successfully.

Answer:"""

_TOOL_ONLY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", _TOOL_ONLY_TEMPLATE),
])


def handle_tool(query: str, tool_name: str):
    """Executes the named tool and formats result via LLM."""
    print(f"  [Handler] Mode: tool ('{tool_name}')")
    session_store.add_user_message(query)
    chat_history = session_store.get_history_text()

    full_response = ""
    try:
        tool_result = _run_tool(tool_name, query)
        llm = get_llm()
        chain = _TOOL_ONLY_PROMPT | llm | StrOutputParser()
        for token in chain.stream({
            "question": query,
            "tool_result": tool_result,
            "chat_history": chat_history
        }):
            full_response += token
            yield token
    except Exception as err:
        print(f"  [Handler Error - Tool] {err}")
        fallback = "I encountered an issue executing that tool. Please try asking directly about specific bus routes or schedules."
        full_response = fallback
        yield fallback
    finally:
        session_store.add_assistant_message(full_response)


# ──────────────────────────────────────────────────────────────────────────────
# Handler 4: RAG + Tool (merged context)
# ──────────────────────────────────────────────────────────────────────────────

_RAG_AND_TOOL_TEMPLATE = """\
Recent Chat History:
{chat_history}

Context from Travel Database:
{context}

Tool Execution Result:
{tool_result}

User Question:
{question}

Instructions:
- Answer using BOTH the database context AND the tool result.
- Keep the response clean, friendly, and structured.

Answer:"""

_RAG_AND_TOOL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", _RAG_AND_TOOL_TEMPLATE),
])


def handle_rag_and_tool(query: str, tool_name: str):
    """Retrieves context AND runs tool, merging both into response."""
    print(f"  [Handler] Mode: rag_and_tool ('{tool_name}')")
    session_store.add_user_message(query)
    chat_history = session_store.get_history_text()

    full_response = ""
    try:
        context = _get_rag_context(query)
        tool_result = _run_tool(tool_name, query)
        llm = get_llm()
        chain = _RAG_AND_TOOL_PROMPT | llm | StrOutputParser()
        for token in chain.stream({
            "question": query,
            "context": context,
            "tool_result": tool_result,
            "chat_history": chat_history,
        }):
            full_response += token
            yield token
    except Exception as err:
        print(f"  [Handler Error - RAG+Tool] {err}")
        yield from handle_rag(query)
    finally:
        if full_response:
            session_store.add_assistant_message(full_response)
