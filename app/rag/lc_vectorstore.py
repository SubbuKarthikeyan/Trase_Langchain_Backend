"""
lc_vectorstore.py
─────────────────
Single source of truth for the LangChain MongoDB Atlas Vector Store connection.
Targets the NEW `bus_routes_v2` collection — the old `bus_routes` is untouched.

Usage
-----
    from app.rag.lc_vectorstore import get_vectorstore
    vs = get_vectorstore()                # returns MongoDBAtlasVectorSearch
    retriever = vs.as_retriever(...)
"""

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

from app.core.config import settings


def _build_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create a Gemini embedding model instance."""
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in environment settings.")

    # gemini-embedding-001 works with the current google-genai SDK.
    # output_dimensionality must match the Atlas vector search index (default 768).
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=settings.EMBEDDING_DIMENSIONS,
    )



@lru_cache(maxsize=1)
def get_vectorstore() -> MongoDBAtlasVectorSearch:
    """
    Returns a cached MongoDBAtlasVectorSearch instance connected to MongoDB.
    """
    if not settings.MONGO_URL:
        raise ValueError("MONGO_URL is not set in .env")

    client = MongoClient(settings.MONGO_URL)
    collection = client[settings.MONGO_DB_NAME][settings.MONGO_COLLECTION_NAME]

    embeddings = _build_embeddings()

    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=settings.MONGO_VECTOR_INDEX,
        text_key="page_content",        # field where document text is stored
        embedding_key="embedding",       # field where embedding vector is stored
        relevance_score_fn="cosine",
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Returns a Gemini embeddings instance configured for document retrieval.
    """
    return _build_embeddings()

