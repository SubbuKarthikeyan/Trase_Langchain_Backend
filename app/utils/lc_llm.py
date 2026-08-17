"""
lc_llm.py
──────────
Provides a LangChain chat LLM with automatic Groq → Gemini fallback.

Usage
-----
    from app.utils.lc_llm import get_llm
    llm = get_llm()      # ChatGroq with Gemini as fallback
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
        Primary:   ChatGroq (configured GROQ_MODEL, e.g. llama-3.3-70b-versatile)
        Fallback1: ChatGroq (llama-3.1-8b-instant)
        Fallback2: ChatGoogleGenerativeAI (configured GEMINI_MODEL, e.g. gemini-2.0-flash)
        Fallback3: ChatGoogleGenerativeAI (gemini-1.5-flash)

    The `with_fallbacks()` wrapper automatically switches models if any exception
    (rate limit 429, 503, quota exhaustion, etc.) occurs.
    """
    fallbacks = []

    primary_model = settings.GROQ_MODEL or "llama-3.3-70b-versatile"
    fallback_groq_model = "llama-3.1-8b-instant"

    if settings.GROQ_API_KEY:
        primary = ChatGroq(
            model=primary_model,
            api_key=settings.GROQ_API_KEY,
            temperature=0.3,
            max_tokens=600,
            streaming=True,
        )
        if fallback_groq_model != primary_model:
            fallbacks.append(
                ChatGroq(
                    model=fallback_groq_model,
                    api_key=settings.GROQ_API_KEY,
                    temperature=0.3,
                    max_tokens=600,
                    streaming=True,
                )
            )
    else:
        primary = None

    gemini_model = settings.GEMINI_MODEL or "gemini-2.0-flash"
    if settings.GEMINI_API_KEY:
        gemini_primary = ChatGoogleGenerativeAI(
            model=gemini_model,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
            max_output_tokens=600,
            streaming=True,
        )
        if primary is None:
            primary = gemini_primary
        else:
            fallbacks.append(gemini_primary)

        if gemini_model != "gemini-1.5-flash":
            fallbacks.append(
                ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=settings.GEMINI_API_KEY,
                    temperature=0.3,
                    max_output_tokens=600,
                    streaming=True,
                )
            )

    if primary is None:
        raise ValueError("Neither GROQ_API_KEY nor GEMINI_API_KEY is configured in settings.")

    return primary.with_fallbacks(fallbacks) if fallbacks else primary
