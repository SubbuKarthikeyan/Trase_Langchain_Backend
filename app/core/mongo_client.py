"""
mongo_client.py
───────────────
Singleton MongoDB client shared across the entire application lifecycle.

Why?
    Previously, every search function (vector, cosine, regex, structured) created
    a new MongoClient per call. Each MongoClient() opens a fresh TCP/TLS connection
    pool to Atlas, adding latency and resource overhead.

    This module provides a single, lazily-initialized MongoClient that is reused
    everywhere — retriever, file_registry, db_init, build_index, etc.

Usage:
    from app.core.mongo_client import get_mongo_client, get_database, get_collection
    client = get_mongo_client()
    db = get_database()
    col = get_collection("bus_routes_v2")
"""

from functools import lru_cache
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from app.core.config import settings


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    """
    Returns a singleton MongoClient instance.
    The client maintains an internal connection pool and is thread-safe.
    """
    if not settings.MONGO_URL:
        raise ValueError("MONGO_URL is not configured in .env")
    return MongoClient(settings.MONGO_URL, serverSelectionTimeoutMS=5000)


def get_database() -> Database:
    """Returns the Trase application database."""
    return get_mongo_client()[settings.MONGO_DB_NAME]


def get_collection(collection_name: str) -> Collection:
    """Returns a specific collection from the Trase database."""
    return get_database()[collection_name]
