"""
file_registry.py
─────────────────
Manages the `file_registry` MongoDB collection — the single source of truth
for tracking which files have been ingested and their MD5 hashes.

Schema per document:
{
    "filename":    str,       # e.g. "bus_data_modified.csv"
    "file_hash":   str,       # MD5 hex digest
    "chunk_count": int,       # number of chunks stored in bus_routes_v2
    "ingested_at": datetime   # UTC timestamp of last successful ingest
}

Why a separate collection?
- Eliminates dependency on LangChain's internal document format for
  change detection (avoids brittle dot-notation reads on the vector store).
- Provides atomicity: registry is only written AFTER chunks are stored.
- Enables fast O(1) lookups instead of full vector-collection scans.
"""

from datetime import datetime, timezone
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection

from app.core.config import settings

REGISTRY_COLLECTION_NAME = getattr(settings, "MONGO_REGISTRY_COLLECTION", "file_registry")


# ──────────────────────────────────────────────────────────────────────────────
# Collection accessor
# ──────────────────────────────────────────────────────────────────────────────

def _get_registry_collection() -> Collection:
    """Returns the raw PyMongo `file_registry` collection."""
    client = MongoClient(settings.MONGO_URL)
    db = client[settings.MONGO_DB_NAME]
    col = db[REGISTRY_COLLECTION_NAME]

    # Ensure a unique index on filename so upserts are safe and fast.
    col.create_index([("filename", ASCENDING)], unique=True, background=True)
    return col


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_all_entries() -> dict[str, str]:
    """
    Returns {filename: file_hash} for every file currently in the registry.
    Used at startup to detect new / modified / deleted files.
    """
    col = _get_registry_collection()
    cursor = col.find({}, {"filename": 1, "file_hash": 1})
    return {doc["filename"]: doc["file_hash"] for doc in cursor}


def upsert_file_entry(filename: str, file_hash: str, chunk_count: int) -> None:
    """
    Insert or update the registry entry for a file.
    Called only after chunks are successfully written to bus_routes_v2.
    """
    col = _get_registry_collection()
    col.update_one(
        {"filename": filename},
        {
            "$set": {
                "filename": filename,
                "file_hash": file_hash,
                "chunk_count": chunk_count,
                "ingested_at": datetime.now(tz=timezone.utc),
            }
        },
        upsert=True,
    )
    print(f"  [Registry] Updated entry for '{filename}' ({chunk_count} chunks).")


def delete_file_entry(filename: str) -> None:
    """
    Removes a file's registry entry when its chunks are pruned from the DB.
    """
    col = _get_registry_collection()
    col.delete_one({"filename": filename})
    print(f"  [Registry] Removed entry for '{filename}'.")


def ensure_vector_source_index() -> None:
    """
    Ensures a MongoDB index on `metadata.source` in the vector store collection.
    Prevents full collection scans when deleting chunks by source filename.
    Called once at startup.
    """
    client = MongoClient(settings.MONGO_URL)
    col = client[settings.MONGO_DB_NAME][settings.MONGO_COLLECTION_NAME]
    col.create_index([("metadata.source", ASCENDING)], background=True)
    print("  [Registry] Index on metadata.source ensured.")
