"""
db_init.py
──────────
Database initialization utility to ensure proper MongoDB collection schemas
and indexes across Trase Bus Travel API startup.
"""

from pymongo import ASCENDING, DESCENDING
from app.core.config import settings
from app.core.mongo_client import get_database


def initialize_database_indexes() -> None:
    """
    Connects to MongoDB and verifies/creates required indexes for all collections:
    - file_registry: unique index on `filename`
    - bus_routes_v2: index on `metadata.source`
    - chat_sessions: unique index on `session_id`, index on `updated_at`
    - structured_bus_routes: unique index on `route_id`, compound index on (`origin`, `destination`)
    """
    if not settings.MONGO_URL:
        print("[DB Init] MONGO_URL not configured. Skipping index initialization.")
        return

    try:
        db = get_database()

        print("=" * 55)
        print(f"[DB Init] Verifying collection indexes in database '{settings.MONGO_DB_NAME}'...")

        # 1. file_registry collection indexes
        registry_col = db[settings.MONGO_REGISTRY_COLLECTION]
        registry_col.create_index([("filename", ASCENDING)], unique=True, background=True)
        print(f"  [DB Init] Collection '{settings.MONGO_REGISTRY_COLLECTION}': Index on 'filename' (unique) verified.")

        # 2. bus_routes_v2 collection indexes
        vector_col = db[settings.MONGO_COLLECTION_NAME]
        vector_col.create_index([("metadata.source", ASCENDING)], background=True)
        print(f"  [DB Init] Collection '{settings.MONGO_COLLECTION_NAME}': Index on 'metadata.source' verified.")

        # 3. chat_sessions collection indexes
        session_col = db[settings.MONGO_SESSION_COLLECTION]
        session_col.create_index([("session_id", ASCENDING)], unique=True, background=True)
        session_col.create_index([("updated_at", DESCENDING)], background=True)
        print(f"  [DB Init] Collection '{settings.MONGO_SESSION_COLLECTION}': Indexes on 'session_id' (unique) & 'updated_at' verified.")

        # 4. structured_bus_routes collection indexes
        routes_col = db[settings.MONGO_STRUCTURED_ROUTES_COLLECTION]
        routes_col.create_index([("route_id", ASCENDING)], unique=True, background=True)
        routes_col.create_index([("origin", ASCENDING), ("destination", ASCENDING)], background=True)
        print(f"  [DB Init] Collection '{settings.MONGO_STRUCTURED_ROUTES_COLLECTION}': Indexes on 'route_id' (unique) & ('origin', 'destination') verified.")

        print("[DB Init] All database schema indexes successfully initialized.")
        print("=" * 55)

    except Exception as err:
        print(f"[DB Init] Warning: Index initialization encountered an issue: {err}")


if __name__ == "__main__":
    initialize_database_indexes()
