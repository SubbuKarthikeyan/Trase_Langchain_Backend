"""
lc_llm.py
──────────
Provides a LangChain chat LLM with automatic Gemini <-> Groq fallback.

Usage
-----
    from app.utils.lc_llm import get_llm
    llm = get_llm()      # ChatGoogleGenerativeAI / ChatGroq with fallback
    llm.stream(messages) # works for streaming
"""

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from app.core.config import settings


@lru_cache(maxsize=1)
def get_llm():
    """
    Returns a LangChain BaseChatModel wired with multi-model fallback:
        Primary:   ChatGoogleGenerativeAI (gemini-3.6-flash)
        Fallback1: ChatGroq (llama-3.1-8b-instant / configured GROQ_MODEL)

    The `with_fallbacks()` wrapper automatically switches models if any exception
    occurs.
    """
    models = []

    # Primary: Gemini 3.6 Flash
    if settings.GEMINI_API_KEY:
        gemini_model = settings.GEMINI_MODEL or "gemini-3.6-flash"
        models.append(
            ChatGoogleGenerativeAI(
                model=gemini_model,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.3,
                max_output_tokens=2048,
                streaming=True,
            )
        )

    # Fallback: Groq LLM
    if settings.GROQ_API_KEY:
        groq_model = settings.GROQ_MODEL or "openai/gpt-oss-20b"
        models.append(
            ChatGroq(
                model=groq_model,
                api_key=settings.GROQ_API_KEY,
                temperature=0.3,
                max_tokens=2048,
                streaming=True,
            )
        )

    if not models:
        raise ValueError("Neither GEMINI_API_KEY nor GROQ_API_KEY is configured in settings.")

    primary = models[0]
    fallbacks = models[1:]
    return primary.with_fallbacks(fallbacks) if fallbacks else primary
