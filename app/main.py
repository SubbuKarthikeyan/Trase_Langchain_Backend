from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import chatbot
from app.routes import query as query_router
from app.auth.routes import router as auth_router
from app.rag.build_index import check_and_update_index
from app.core.db_init import initialize_database_indexes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts.
    1. Initializes collection indexes across MongoDB.
    2. Scans data/raw directory, checks file_registry for existing hashes,
       and ingests only new or modified documents into bus_routes_v2.
    """
    initialize_database_indexes()
    check_and_update_index()

    yield  # Server is now running

    print("Server shutting down.")


app = FastAPI(
    title="Trase Bus Travel API",
    version="2.0.0",
    lifespan=lifespan,
)

# Allow React dev server with credentials for cookie-based refresh tokens
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def landing_page():
    return {
        "message": "Trase Bus Travel API",
        "endpoints": {
            "auth": "/api/auth",
            "smart_chat": "POST /query/stream  (JSON body: {message: '...'})",
            "legacy_chat": "POST /chatbot/stream?message=...",
            "router_health": "GET /query/health",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "message": "Backend is healthy."}


# Authentication router
app.include_router(auth_router)

# Legacy RAG-only router (preserved for backward compatibility)
app.include_router(chatbot.router)

# New intelligent query router (routes to GeneralLLM / RAG / Tool / RAG+Tool)
app.include_router(query_router.router)

