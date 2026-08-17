"""
chatbot.py
───────────
FastAPI router for the Trase Bus Travel chatbot.

Streaming endpoint uses a LangChain LCEL pipeline:
    retriever  →  format_docs  →  RAG_PROMPT  →  LLM (Groq → Gemini fallback)  →  StrOutputParser
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from app.rag.retriever import Retriever
from app.utils.lc_llm import get_llm
from app.prompts.rag_prompt import RAG_PROMPT

router = APIRouter()
_retriever = Retriever(top_k=5)


import sys

def _get_context(question: str) -> str:
    """Retrieves matching documents from database/vectorstore with logging & context verification."""
    chunks = _retriever.retrieve(question)
    print(f"\n[Database Context Verification] Retrieved {len(chunks)} document context(s).")
    for idx, chunk in enumerate(chunks, 1):
        print(f"--- Doc Chunk {idx} ---")
        try:
            print(chunk)
        except UnicodeEncodeError:
            print(chunk.encode('ascii', errors='replace').decode('ascii'))
    print("-" * 55)
    return "\n\n".join(chunks)



def _build_chain():
    llm = get_llm()

    chain = (
        {
            "context": RunnableLambda(_get_context),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


_chain = _build_chain()



# ──────────────────────────────────────────────────────────────────────────────
# Streaming generator
# ──────────────────────────────────────────────────────────────────────────────

def stream_generator(message: str):
    print("=" * 55)
    print("[LangChain Pipeline] Request received:", message)
    try:
        chain = _build_chain()
        for token in chain.stream(message):
            yield token
    except Exception as err:
        print(f"[LangChain Pipeline Error] {err}")
        try:
            raw_context = _get_context(message)
            if raw_context:
                yield "Here are the details from our database:\n\n" + raw_context
            else:
                yield "I'm sorry, but those details are not available in our travel database at the moment."
        except Exception:
            yield "I'm experiencing high server demand right now. Please try again in a few moments."



# ──────────────────────────────────────────────────────────────────────────────
# Route
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/chatbot/stream")
def chatbot_stream(message: str):
    """
    Streams an AI response for the given message using RAG over bus_routes_v2.
    """
    return StreamingResponse(
        stream_generator(message),
        media_type="text/plain",
    )
