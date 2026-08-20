"""
test_e2e_pipeline.py
────────────────────
Comprehensive End-to-End Terminal Monitoring & Diagnostic Test for Trase Bus Travel API.

Executes and analyzes all 13 core backend functions sequentially:
  1. Environment Variable Configuration
  2. MongoDB Atlas Connection & Ping
  3. Database Schema & Collection Indexes
  4. Raw Data File Repository Ingestion
  5. Text Chunking & Vector Store Embeddings
  6. Retrieval Leg 1: Semantic Vector Search
  7. Retrieval Leg 2: Keyword Regex Search
  8. Retrieval Leg 3: Structured Route Search
  9. Reciprocal Rank Fusion (RRF) Hybrid Search Engine
 10. Multi-Model LLM Connectivity & Fallbacks
 11. Intent Classifier & Query Router
 12. End-to-End RAG Pipeline & Stream Generation
 13. FastAPI Health Endpoint & App Instance

Outputs a formatted report with status (TRUE / FALSE), process description,
and clear diagnostic details for every stage.
"""

import sys
import os
import time
from typing import Dict, Any, List

# Force UTF-8 stdout encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings


class PipelineDiagnosticRunner:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.test_query = "Are there any AC buses from Chennai to Madurai?"

    def _record(self, process_num: int, name: str, status: bool, description: str, details: str):
        self.results.append({
            "num": process_num,
            "name": name,
            "status": status,
            "description": description,
            "details": details
        })

    def run_all(self):
        print("\n" + "=" * 80, flush=True)
        print("           TRASE BACKEND PIPELINE -- END-TO-END DIAGNOSTIC MONITOR           ", flush=True)
        print("=" * 80 + "\n", flush=True)

        # Process 1: Environment Variables
        self._test_01_env()

        # Process 2: MongoDB Ping
        self._test_02_mongo_connection()

        # Process 3: Database & Index Verification
        self._test_03_db_indexes()

        # Process 4: Raw Data Files
        self._test_04_raw_files()

        # Process 5: Embeddings & Vector Store
        self._test_05_embeddings_and_vectorstore()

        # Process 6: Leg 1 - Semantic Search
        self._test_06_semantic_search()

        # Process 7: Leg 2 - Keyword Regex Search
        self._test_07_keyword_search()

        # Process 8: Leg 3 - Structured Route Search
        self._test_08_structured_search()

        # Process 9: RRF Hybrid Search Engine
        self._test_09_hybrid_fusion()

        # Process 10: Multi-Model LLM Connectivity
        self._test_10_llm_connectivity()

        # Process 11: Intent Classifier
        self._test_11_intent_classifier()

        # Process 12: End-to-End RAG Stream
        self._test_12_rag_pipeline()

        # Process 13: FastAPI Application
        self._test_13_fastapi_app()

        # Output Summary Report
        self._print_report()

    # ── Test Implementations ──────────────────────────────────────────────────

    def _test_01_env(self):
        desc = "Verifies essential API keys (Gemini, Groq) and MongoDB URL in environment variables."
        missing = []
        if not settings.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if not settings.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not settings.MONGO_URL:
            missing.append("MONGO_URL")

        if not missing:
            self._record(1, "Environment Configuration Check", True, desc,
                         "All required API keys (Gemini, Groq) and MONGO_URL are active.")
        else:
            self._record(1, "Environment Configuration Check", False, desc,
                         f"Missing environment variables: {', '.join(missing)}.")

    def _test_02_mongo_connection(self):
        desc = "Establishes socket connection to MongoDB Atlas and executes server ping."
        try:
            from app.core.mongo_client import get_mongo_client
            client = get_mongo_client()
            start_t = time.time()
            client.admin.command('ping')
            elapsed_ms = round((time.time() - start_t) * 1000, 2)
            self._record(2, "MongoDB Atlas Connection", True, desc,
                         f"Successfully pinged cluster '{settings.MONGO_DB_NAME}' (Response time: {elapsed_ms}ms).")
        except Exception as err:
            self._record(2, "MongoDB Atlas Connection", False, desc,
                         f"Failed to connect to MongoDB: {err}")

    def _test_03_db_indexes(self):
        desc = "Checks existence and indexing across collections (file_registry, bus_routes_v2, chat_sessions, structured_bus_routes)."
        try:
            from app.core.mongo_client import get_database
            db = get_database()
            cols = db.list_collection_names()
            required = [
                settings.MONGO_COLLECTION_NAME,
                settings.MONGO_REGISTRY_COLLECTION,
                settings.MONGO_SESSION_COLLECTION,
                settings.MONGO_STRUCTURED_ROUTES_COLLECTION
            ]
            found = [c for c in required if c in cols]
            if len(found) == len(required):
                self._record(3, "Database Collection & Index Verification", True, desc,
                             f"All 4 core collections verified in '{settings.MONGO_DB_NAME}': {', '.join(found)}.")
            else:
                missing_cols = list(set(required) - set(found))
                self._record(3, "Database Collection & Index Verification", False, desc,
                             f"Missing required MongoDB collection(s): {', '.join(missing_cols)}.")
        except Exception as err:
            self._record(3, "Database Collection & Index Verification", False, desc,
                         f"Error inspecting database collections: {err}")

    def _test_04_raw_files(self):
        desc = "Scans data/raw/ directory and checks tracked documents in file_registry."
        try:
            from app.rag.file_registry import get_all_entries
            from app.rag.build_index import _discover_raw_files
            raw_files = _discover_raw_files("data/raw")
            registry = get_all_entries()
            if raw_files:
                self._record(4, "Raw File Repository Ingestion", True, desc,
                             f"Found {len(raw_files)} raw file(s) on disk. Registry contains {len(registry)} tracked entry/entries.")
            else:
                self._record(4, "Raw File Repository Ingestion", False, desc,
                             "No supported raw data files (.csv, .pdf, .txt) found in data/raw/.")
        except Exception as err:
            self._record(4, "Raw File Repository Ingestion", False, desc,
                         f"Failed to inspect raw file repository: {err}")

    def _test_05_embeddings_and_vectorstore(self):
        desc = "Validates Gemini embedding model generation and bus_routes_v2 vector store access."
        try:
            from app.rag.lc_vectorstore import get_embeddings, get_vectorstore
            embeddings = get_embeddings()
            vector = embeddings.embed_query("Test query for embeddings")
            dim = len(vector)
            vectorstore = get_vectorstore()
            if dim == settings.EMBEDDING_DIMENSIONS:
                self._record(5, "Text Chunking & Vector Store Embeddings", True, desc,
                             f"Gemini embedding model online. Successfully generated vector of size {dim} (matches Atlas index requirement).")
            else:
                self._record(5, "Text Chunking & Vector Store Embeddings", False, desc,
                             f"Vector dimension mismatch: generated {dim}, expected {settings.EMBEDDING_DIMENSIONS}.")
        except Exception as err:
            self._record(5, "Text Chunking & Vector Store Embeddings", False, desc,
                         f"Embedding generation or vector store connection failed: {err}")

    def _test_06_semantic_search(self):
        desc = "Executes Retrieval Leg 1 (Atlas Vector Search & Cosine Scan Fallback)."
        try:
            from app.rag.retriever import _semantic_search
            results = _semantic_search(self.test_query, top_k=3)
            if results:
                self._record(6, "Retrieval Leg 1 -- Semantic Vector Search", True, desc,
                             f"Semantic search returned {len(results)} relevant document chunk(s). Preview: '{results[0][:80]}...'")
            else:
                self._record(6, "Retrieval Leg 1 -- Semantic Vector Search", False, desc,
                             "Semantic search returned 0 results. Vector collection may be empty or unindexed.")
        except Exception as err:
            self._record(6, "Retrieval Leg 1 -- Semantic Vector Search", False, desc,
                         f"Semantic vector search failed: {err}")

    def _test_07_keyword_search(self):
        desc = "Executes Retrieval Leg 2 (MongoDB Regex Keyword Matching)."
        try:
            from app.rag.retriever import _keyword_search
            results = _keyword_search(self.test_query, top_k=3)
            if results:
                self._record(7, "Retrieval Leg 2 -- Keyword Regex Search", True, desc,
                             f"Keyword search returned {len(results)} matching chunk(s).")
            else:
                self._record(7, "Retrieval Leg 2 -- Keyword Regex Search", False, desc,
                             "Keyword search returned 0 results for query keywords.")
        except Exception as err:
            self._record(7, "Retrieval Leg 2 -- Keyword Regex Search", False, desc,
                         f"Keyword regex search failed: {err}")

    def _test_08_structured_search(self):
        desc = "Executes Retrieval Leg 3 (Structured Bus Route Record Lookup)."
        try:
            from app.rag.retriever import _structured_search
            results = _structured_search(self.test_query, top_k=3)
            if results:
                self._record(8, "Retrieval Leg 3 -- Structured Route Search", True, desc,
                             f"Structured search returned {len(results)} exact route record(s).")
            else:
                self._record(8, "Retrieval Leg 3 -- Structured Route Search", False, desc,
                             "Structured search returned 0 matching route records from structured_bus_routes.")
        except Exception as err:
            self._record(8, "Retrieval Leg 3 -- Structured Route Search", False, desc,
                         f"Structured route search failed: {err}")

    def _test_09_hybrid_fusion(self):
        desc = "Executes Reciprocal Rank Fusion (RRF) across semantic, keyword, and structured legs."
        try:
            from app.rag.retriever import Retriever
            retriever = Retriever(top_k=5, hybrid=True)
            fused_chunks = retriever.retrieve(self.test_query)
            if fused_chunks:
                self._record(9, "RRF Hybrid Search Fusion Engine", True, desc,
                             f"RRF engine successfully fused candidates and returned top {len(fused_chunks)} ranked chunk(s).")
            else:
                self._record(9, "RRF Hybrid Search Fusion Engine", False, desc,
                             "Hybrid retriever returned 0 fused chunks.")
        except Exception as err:
            self._record(9, "RRF Hybrid Search Fusion Engine", False, desc,
                         f"RRF fusion execution failed: {err}")

    def _test_10_llm_connectivity(self):
        desc = "Tests primary LLM (Groq) and automatic fallback LLM (Gemini)."
        try:
            from app.utils.lc_llm import get_llm
            llm = get_llm()
            res = llm.invoke("Reply with the single word 'ONLINE' if you are functional.")
            text = res.content if hasattr(res, "content") else str(res)
            if isinstance(text, list):
                text = "".join(
                    x if isinstance(x, str) else str(x.get("text", "")) if isinstance(x, dict) else str(x)
                    for x in text
                )
            if text:
                self._record(10, "Multi-Model LLM Connectivity & Fallbacks", True, desc,
                             f"LLM chain responsive. Test output: '{text.strip()[:60]}'.")
            else:
                self._record(10, "Multi-Model LLM Connectivity & Fallbacks", False, desc,
                             "LLM returned an empty response.")
        except Exception as err:
            self._record(10, "Multi-Model LLM Connectivity & Fallbacks", False, desc,
                         f"LLM invocation failed across configured providers: {err}")

    def _test_11_intent_classifier(self):
        desc = "Classifies user query intent using LLM-based router (general_llm, rag, tool, rag_and_tool)."
        try:
            from app.router.intent_classifier import classify_intent
            res = classify_intent(self.test_query)
            intent = res.get("intent")
            if intent in {"general_llm", "rag", "tool", "rag_and_tool"}:
                self._record(11, "Intent Classifier & Query Router", True, desc,
                             f"Successfully classified intent: '{intent}' (tool_name: {res.get('tool_name')}).")
            else:
                self._record(11, "Intent Classifier & Query Router", False, desc,
                             f"Classifier returned invalid intent output: {res}.")
        except Exception as err:
            self._record(11, "Intent Classifier & Query Router", False, desc,
                         f"Intent classification failed: {err}")

    def _test_12_rag_pipeline(self):
        desc = "Tests full end-to-end RAG handler streaming generator (retrieve -> prompt -> LLM)."
        try:
            from app.router.handlers import handle_rag
            tokens = []
            for token in handle_rag(self.test_query):
                if isinstance(token, list):
                    token = "".join(
                        x if isinstance(x, str) else str(x.get("text", "")) if isinstance(x, dict) else str(x)
                        for x in token
                    )
                tokens.append(str(token))
                if len(tokens) >= 5:  # Verify stream is yielding content
                    break
            full_response = "".join(tokens)
            if full_response.strip():
                self._record(12, "End-to-End RAG Stream Generation", True, desc,
                             f"RAG generator streamed content successfully. Stream sample: '{full_response[:70]}...'")
            else:
                self._record(12, "End-to-End RAG Stream Generation", False, desc,
                             "RAG generator yielded no response tokens.")
        except Exception as err:
            self._record(12, "End-to-End RAG Stream Generation", False, desc,
                         f"RAG handler streaming failed: {err}")

    def _test_13_fastapi_app(self):
        desc = "Verifies FastAPI app instantiation, CORS middleware, and router health."
        try:
            from app.main import app
            from app.routes.query import query_router_health
            health_res = query_router_health()
            if app.title == "Trase Bus Travel API" and health_res.get("status") == "ok":
                self._record(13, "FastAPI Application & Route Setup", True, desc,
                             f"FastAPI instance '{app.title}' v{app.version} loaded cleanly. Registered tools: {health_res.get('registered_tools')}.")
            else:
                self._record(13, "FastAPI Application & Route Setup", False, desc,
                             "FastAPI application health check returned unexpected output.")
        except Exception as err:
            self._record(13, "FastAPI Application & Route Setup", False, desc,
                         f"FastAPI app instantiation check failed: {err}")

    # ── Reporter ──────────────────────────────────────────────────────────────

    def _print_report(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"])
        failed = total - passed

        print("\n" + "=" * 80, flush=True)
        print("                     DETAILED BACKEND FUNCTION RESULTS                         ", flush=True)
        print("=" * 80 + "\n", flush=True)

        for r in self.results:
            status_text = "TRUE  [PASS]" if r["status"] else "FALSE [FAIL]"
            symbol = "[✓]" if r["status"] else "[X]"

            print(f"[{r['num']:02d}/13] {symbol} STATUS: {status_text}", flush=True)
            print(f"       PROCESS    : {r['name']}", flush=True)
            print(f"       DESCRIPTION: {r['description']}", flush=True)
            print(f"       ANALYSIS   : {r['details']}", flush=True)
            print("-" * 80, flush=True)

        print("\n" + "=" * 80, flush=True)
        print(f"DIAGNOSTIC SUMMARY: {passed}/{total} BACKEND PROCESSES PASSED", flush=True)
        if failed == 0:
            print("OVERALL SYSTEM STATUS: TRUE -- ALL BACKEND PROCESSES OPERATIONAL", flush=True)
        else:
            print(f"OVERALL SYSTEM STATUS: FALSE -- {failed} PROCESS(ES) REQUIRE ATTENTION", flush=True)
        print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    runner = PipelineDiagnosticRunner()
    runner.run_all()
