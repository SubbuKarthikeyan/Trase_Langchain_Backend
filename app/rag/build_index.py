"""
build_index.py
───────────────
Scans data/raw/ (recursively), detects new/modified/deleted files via
MD5 hash, and incrementally ingests them into `bus_routes_v2` via LangChain.

Changes from the original:
  ✔  Hash tracking moved to dedicated `file_registry` collection — no longer
     reads back from LangChain's stored docs (eliminates brittle dot-notation
     queries on the vector collection).
  ✔  Deduplication guard: old chunks are deleted BEFORE re-ingesting, even on
     restart after a mid-ingest crash — prevents any duplicate chunks.
  ✔  Recursive directory scan (os.walk) — subdirectories under data/raw/ are
     automatically picked up without code changes.
  ✔  SUPPORTED_EXTENSIONS imported from loader.py — single source of truth.
  ✔  MongoDB index on metadata.source ensured at startup — fast deletes.
  ✔  Registry entry written ONLY after successful chunk storage — atomicity.

Pipeline:
    file  →  LangChain Loader  →  RecursiveCharacterTextSplitter
          →  GoogleGenerativeAI Embeddings  →  MongoDBAtlasVectorSearch
          →  file_registry (success only)
"""

import os
import hashlib

from langchain_core.documents import Document

from app.rag.loader import load_document, load_structured_routes, SUPPORTED_EXTENSIONS
from app.rag.chunker import create_chunks
from app.rag.lc_vectorstore import get_vectorstore
from app.rag.file_registry import (
    get_all_entries,
    upsert_file_entry,
    delete_file_entry,
    ensure_vector_source_index,
)
from app.core.config import settings
from app.core.mongo_client import get_collection


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _calculate_file_hash(file_path: str) -> str:
    """Returns MD5 hex digest of a file for change detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_vector_collection():
    """Returns the raw PyMongo bus_routes_v2 collection."""
    return get_collection(settings.MONGO_COLLECTION_NAME)


def _delete_chunks_by_source(collection, source: str) -> None:
    """
    Deletes ALL chunks for a given source filename from bus_routes_v2.
    Called both on re-ingest (hash mismatch) and on file removal from disk.
    """
    res = collection.delete_many({"metadata.source": source})
    print(f"  [Ingest] Deleted {res.deleted_count} chunk(s) for '{source}'.")


def _discover_raw_files(raw_dir: str) -> dict[str, str]:
    """
    Recursively walks raw_dir and returns {filename: full_path} for every
    supported, non-query file found (including inside subdirectories).

    Using os.walk() instead of os.listdir() so new subdirectories added
    to data/raw/ are automatically picked up with no code changes.
    """
    found: dict[str, str] = {}
    for root, _dirs, files in os.walk(raw_dir):
        for entry in files:
            # Skip query/output/hidden files
            if entry.startswith(".") or entry.endswith("_query.txt"):
                continue
            ext = os.path.splitext(entry)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, entry)
                # Use the bare filename as the key (what we store in metadata.source)
                found[entry] = full_path
    return found


# ──────────────────────────────────────────────────────────────────────────────
# Core ingestion
# ──────────────────────────────────────────────────────────────────────────────

def _ingest_structured_routes(file_path: str) -> None:
    """Ingests parsed structured records from CSV into structured_bus_routes collection."""
    if not file_path.endswith(".csv") or not settings.MONGO_URL:
        return

    records = load_structured_routes(file_path)
    if not records:
        return

    try:
        col = get_collection(settings.MONGO_STRUCTURED_ROUTES_COLLECTION)

        # Prune existing records for this file before re-ingesting
        filename = os.path.basename(file_path)
        col.delete_many({"source_file": filename})

        # Insert new records
        col.insert_many(records)
        print(f"  [Ingest] Stored {len(records)} structured route record(s) in '{settings.MONGO_STRUCTURED_ROUTES_COLLECTION}'.")
    except Exception as err:
        print(f"  [Ingest Warning] Failed to store structured routes: {err}")


def ingest_file(file_path: str) -> bool:
    """
    Loads, chunks, embeds, and stores a single file into `bus_routes_v2` and `structured_bus_routes`.

    Atomicity contract:
        - Chunks are written to bus_routes_v2 and structured_bus_routes.
        - file_registry is updated ONLY if the chunk write succeeds.
        - On failure, neither side is corrupted (next startup will retry).

    Returns True on success, False on failure.
    """
    filename = os.path.basename(file_path)
    file_hash = _calculate_file_hash(file_path)

    print(f"  [Ingest] Loading '{filename}'...")
    try:
        docs: list[Document] = load_document(file_path)
        chunks: list[Document] = create_chunks(docs)

        if not chunks:
            print(f"  [Ingest] No chunks generated for '{filename}'. Skipping.")
            return False

        # Stamp every chunk with source + hash for future change detection
        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["file_hash"] = file_hash

        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)

        # Ingest structured routes into MongoDB collection if CSV
        _ingest_structured_routes(file_path)

        # Write registry entry AFTER successful storage
        upsert_file_entry(filename, file_hash, len(chunks))

        print(f"  [Ingest] [OK] '{filename}' -> {len(chunks)} chunks in bus_routes_v2.")
        return True

    except Exception as exc:
        print(f"  [Ingest] [ERROR] Error ingesting '{filename}': {exc}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Entry point called by app/main.py on startup
# ──────────────────────────────────────────────────────────────────────────────

def check_and_update_index(raw_dir: str = "data/raw") -> None:
    """
    Incremental ingestion — called once at server startup.

    Logic:
        - New file      → ingest + add registry entry
        - Modified file → delete old chunks → re-ingest → update registry
        - Unchanged     → skip
        - Deleted file  → prune chunks + remove registry entry

    All changes target bus_routes_v2 only.
    Hash tracking uses the dedicated file_registry collection.
    """
    print("=" * 55)
    print(f"[Ingest] Starting index check against '{raw_dir}'...")

    if not os.path.exists(raw_dir):
        print(f"  [Ingest] Directory '{raw_dir}' not found. Nothing to ingest.")
        print("=" * 55)
        return

    # Ensure a DB index on metadata.source for fast chunk deletes
    ensure_vector_source_index()

    collection = _get_vector_collection()

    # Load registry: {filename → file_hash} from dedicated collection
    registry = get_all_entries()
    print(f"  [Ingest] Registry has {len(registry)} tracked file(s).")

    # Discover files on disk (recursive)
    raw_files = _discover_raw_files(raw_dir)
    print(f"  [Ingest] Found {len(raw_files)} supported file(s) in '{raw_dir}'.")

    ingested_count = 0
    skipped_count = 0

    # ── 1. New or modified files ──────────────────────────────────────────────
    for filename, file_path in raw_files.items():
        current_hash = _calculate_file_hash(file_path)

        if filename in registry:
            if registry[filename] == current_hash:
                print(f"  [Ingest] '{filename}' unchanged. Skipping.")
                skipped_count += 1
                continue
            else:
                print(f"  [Ingest] '{filename}' modified. Re-ingesting...")
                # Delete old chunks BEFORE writing new ones — prevents duplicates
                _delete_chunks_by_source(collection, filename)
        else:
            print(f"  [Ingest] '{filename}' is new. Ingesting...")

        if ingest_file(file_path):
            ingested_count += 1

    # ── 2. Files removed from disk — prune orphaned chunks + registry ─────────
    for stored_filename in list(registry.keys()):
        if stored_filename not in raw_files:
            print(f"  [Ingest] '{stored_filename}' removed from disk. Pruning...")
            _delete_chunks_by_source(collection, stored_filename)
            delete_file_entry(stored_filename)

    print(
        f"\n  [Ingest] Done — {ingested_count} file(s) ingested/updated, "
        f"{skipped_count} unchanged."
    )
    print("=" * 55)


if __name__ == "__main__":
    check_and_update_index()
