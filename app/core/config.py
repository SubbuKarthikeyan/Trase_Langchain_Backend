from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MONGO_URL = os.getenv("MONGO_URL")

    # Canonical Mongo database & vector store settings
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "trase_bus_db")
    MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME_V2", "bus_routes_v2")
    MONGO_VECTOR_INDEX = os.getenv("MONGO_VECTOR_INDEX_V2", "vector_index_v2")

    # Additional collection names
    MONGO_REGISTRY_COLLECTION = os.getenv("MONGO_REGISTRY_COLLECTION", "file_registry")
    MONGO_SESSION_COLLECTION = os.getenv("MONGO_SESSION_COLLECTION", "chat_sessions")
    MONGO_STRUCTURED_ROUTES_COLLECTION = os.getenv("MONGO_STRUCTURED_ROUTES_COLLECTION", "structured_bus_routes")

    # Embedding model (must match MongoDB Atlas vector index dimensions)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

    # LLM Models & Fallback
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    # Hybrid Search / RRF settings
    RRF_K = int(os.getenv("RRF_K", "60"))
    HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"

    # Backwards-compatibility aliases
    MONGO_COLLECTION_NAME_V2 = MONGO_COLLECTION_NAME
    MONGO_VECTOR_INDEX_V2 = MONGO_VECTOR_INDEX

    # AgentMail Settings
    EMAIL_SENDER = os.getenv("EMAIL_SENDER", "trase_mail@agentmail.to")
    AGENTMAIL_API_KEY = os.getenv("AGENTMAIL_API_KEY")

    # ── Auth / JWT Settings ──────────────────────────────────────────────────
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
    JWT_REFRESH_SECRET_KEY = os.getenv("JWT_REFRESH_SECRET_KEY", "change-me-refresh")
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    # ── Clerk (backend verification) ─────────────────────────────────────────
    CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
    CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")

    # ── Auth MongoDB Collections ─────────────────────────────────────────────
    MONGO_USERS_COLLECTION = os.getenv("MONGO_USERS_COLLECTION", "users")
    MONGO_PROFILES_COLLECTION = os.getenv("MONGO_PROFILES_COLLECTION", "profiles")
    MONGO_CONNECTORS_COLLECTION = os.getenv("MONGO_CONNECTORS_COLLECTION", "connectors")


settings = Settings()
