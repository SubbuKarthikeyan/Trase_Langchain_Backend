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

    # TollCall API Keys
    EMAIL_SENDER=am_us_inbox_7b9c3dbb5b2be5ffa6e57b58e6ab7b089e747fbe364dbd37383454ca7400b506
    

settings = Settings()



