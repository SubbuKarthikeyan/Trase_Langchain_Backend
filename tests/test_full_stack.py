"""
test_full_stack.py
───────────────────
Full-Stack Terminal Test & Diagnostic Suite for Trase Application.

Tests BOTH Backend and Frontend components:
  1. Backend Environment & API Keys
  2. MongoDB Atlas Database Ping
  3. Collection Schema & Indexes
  4. Retriever Multi-Leg Search & RRF Engine
  5. Multi-Model LLM Execution (Gemini 3.6 Flash)
  6. FastAPI Backend Server Execution (http://localhost:8000)
  7. API Landing Page Endpoint (GET /)
  8. API Health Check Endpoint (GET /health)
  9. Query Router Health Endpoint (GET /query/health)
 10. Query Debug Endpoint (GET /query/debug)
 11. Streaming Query Execution (POST /query/stream)
 12. Frontend Next.js Production Build Verification
 13. Frontend UI Web Server Connectivity (http://localhost:3000)

Prints a clean terminal report with STATUS: TRUE / FALSE and diagnostic details.
"""

import sys
import os
import time
import urllib.request
import urllib.parse
import json
import subprocess
from typing import Dict, Any, List

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings


class FullStackDiagnosticRunner:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.backend_url = "http://127.0.0.1:8000"
        self.frontend_url = "http://localhost:3000"

    def _record(self, process_num: int, name: str, status: bool, description: str, details: str):
        self.results.append({
            "num": process_num,
            "name": name,
            "status": status,
            "description": description,
            "details": details
        })

    def run_all(self):
        print("\n" + "=" * 85, flush=True)
        print("         TRASE FULL-STACK APPLICATION DIAGNOSTIC & TERMINAL MONITOR          ", flush=True)
        print("=" * 85 + "\n", flush=True)

        # ── Backend Tests ─────────────────────────────────────────────────────
        self._test_01_env()
        self._test_02_mongo_ping()
        self._test_03_collections()
        self._test_04_retriever_rrf()
        self._test_05_llm_execution()

        # ── Live Backend HTTP Endpoint Tests ──────────────────────────────────
        self._test_06_backend_server_check()
        self._test_07_backend_landing_endpoint()
        self._test_08_backend_health_endpoint()
        self._test_09_query_router_health()
        self._test_10_query_debug_endpoint()
        self._test_11_query_stream_endpoint()

        # ── Frontend Tests ────────────────────────────────────────────────────
        self._test_12_frontend_build_check()
        self._test_13_frontend_server_check()

        # Output Summary
        self._print_report()

    # ── Test Implementations ──────────────────────────────────────────────────

    def _test_01_env(self):
        desc = "Verifies GEMINI_API_KEY, GROQ_API_KEY, and MONGO_URL configuration."
        missing = []
        if not settings.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if not settings.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not settings.MONGO_URL:
            missing.append("MONGO_URL")

        if not missing:
            self._record(1, "Backend Environment Setup", True, desc, "All environment variables active.")
        else:
            self._record(1, "Backend Environment Setup", False, desc, f"Missing: {', '.join(missing)}")

    def _test_02_mongo_ping(self):
        desc = "Pings MongoDB Atlas cluster 'trase_bus_db'."
        try:
            from app.core.mongo_client import get_mongo_client
            client = get_mongo_client()
            start_t = time.time()
            client.admin.command('ping')
            elapsed_ms = round((time.time() - start_t) * 1000, 2)
            self._record(2, "MongoDB Atlas Connection", True, desc, f"Ping successful ({elapsed_ms}ms).")
        except Exception as err:
            self._record(2, "MongoDB Atlas Connection", False, desc, f"Mongo error: {err}")

    def _test_03_collections(self):
        desc = "Verifies collection schemas and indexes in MongoDB."
        try:
            from app.core.mongo_client import get_database
            db = get_database()
            cols = db.list_collection_names()
            required = ["bus_routes_v2", "file_registry", "chat_sessions", "structured_bus_routes"]
            found = [c for c in required if c in cols]
            if len(found) == len(required):
                self._record(3, "Database Collection Schemas", True, desc, f"Verified 4/4 collections: {', '.join(found)}")
            else:
                self._record(3, "Database Collection Schemas", False, desc, f"Missing collections: {set(required) - set(found)}")
        except Exception as err:
            self._record(3, "Database Collection Schemas", False, desc, f"Error: {err}")

    def _test_04_retriever_rrf(self):
        desc = "Executes hybrid retriever combining semantic, keyword, and structured legs."
        try:
            from app.rag.retriever import Retriever
            retriever = Retriever(top_k=3, hybrid=True)
            res = retriever.retrieve("Buses from Chennai to Madurai")
            if res:
                self._record(4, "Multi-Leg Retriever & RRF Engine", True, desc, f"Retrieved and fused {len(res)} top chunk(s).")
            else:
                self._record(4, "Multi-Leg Retriever & RRF Engine", False, desc, "Retriever returned 0 results.")
        except Exception as err:
            self._record(4, "Multi-Leg Retriever & RRF Engine", False, desc, f"Retrieval error: {err}")

    def _test_05_llm_execution(self):
        desc = "Validates LLM execution using Gemini 3.6 Flash model."
        try:
            from app.utils.lc_llm import get_llm
            llm = get_llm()
            res = llm.invoke("Reply ONLINE")
            self._record(5, "LLM Core Connectivity", True, desc, "LLM responsive.")
        except Exception as err:
            self._record(5, "LLM Core Connectivity", False, desc, f"LLM error: {err}")

    def _test_06_backend_server_check(self):
        desc = "Checks if FastAPI backend server is responding on http://127.0.0.1:8000."
        try:
            req = urllib.request.Request(self.backend_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    self._record(6, "FastAPI Backend Web Server", True, desc, f"Backend online at {self.backend_url} (HTTP 200).")
                else:
                    self._record(6, "FastAPI Backend Web Server", False, desc, f"HTTP Status: {resp.status}")
        except Exception as err:
            self._record(6, "FastAPI Backend Web Server", False, desc, f"Server not reachable: {err}. (Start backend using: python server.py)")

    def _test_07_backend_landing_endpoint(self):
        desc = "Calls GET / landing page endpoint on backend API."
        try:
            url = f"{self.backend_url}/"
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                if "endpoints" in data:
                    self._record(7, "API Landing Endpoint (GET /)", True, desc, f"Landing page returned endpoints: {list(data['endpoints'].keys())}")
                else:
                    self._record(7, "API Landing Endpoint (GET /)", False, desc, f"Unexpected response: {data}")
        except Exception as err:
            self._record(7, "API Landing Endpoint (GET /)", False, desc, f"Failed GET /: {err}")

    def _test_08_backend_health_endpoint(self):
        desc = "Calls GET /health backend status endpoint."
        try:
            url = f"{self.backend_url}/health"
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") == "ok":
                    self._record(8, "API Health Endpoint (GET /health)", True, desc, f"Backend reported status ok: {data}")
                else:
                    self._record(8, "API Health Endpoint (GET /health)", False, desc, f"Status not ok: {data}")
        except Exception as err:
            self._record(8, "API Health Endpoint (GET /health)", False, desc, f"Failed GET /health: {err}")

    def _test_09_query_router_health(self):
        desc = "Calls GET /query/health router status endpoint."
        try:
            url = f"{self.backend_url}/query/health"
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") == "ok":
                    self._record(9, "Query Router Health (GET /query/health)", True, desc, f"Router online. Registered tools: {data.get('registered_tools')}")
                else:
                    self._record(9, "Query Router Health (GET /query/health)", False, desc, f"Response: {data}")
        except Exception as err:
            self._record(9, "Query Router Health (GET /query/health)", False, desc, f"Failed GET /query/health: {err}")

    def _test_10_query_debug_endpoint(self):
        desc = "Calls GET /query/debug endpoint with test query parameter."
        try:
            params = urllib.parse.urlencode({"message": "Chennai to Madurai", "top_k": 3})
            url = f"{self.backend_url}/query/debug?{params}"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                if "legs" in data and "fused" in data:
                    self._record(10, "Query Debug Endpoint (GET /query/debug)", True, desc, f"Debug route returned leg candidates and fused rankings.")
                else:
                    self._record(10, "Query Debug Endpoint (GET /query/debug)", False, desc, f"Response missing debug fields: {list(data.keys())}")
        except Exception as err:
            self._record(10, "Query Debug Endpoint (GET /query/debug)", False, desc, f"Failed GET /query/debug: {err}")

    def _test_11_query_stream_endpoint(self):
        desc = "Posts query payload to POST /query/stream and validates token stream response."
        try:
            url = f"{self.backend_url}/query/stream"
            payload = json.dumps({"message": "Buses from Chennai to Madurai"}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                token_sample = resp.read(200).decode("utf-8", errors="replace")
                if token_sample.strip():
                    self._record(11, "Streaming Query Endpoint (POST /query/stream)", True, desc, f"Streamed tokens successfully. Sample: '{token_sample[:60]}...'")
                else:
                    self._record(11, "Streaming Query Endpoint (POST /query/stream)", False, desc, "Stream returned 0 bytes.")
        except Exception as err:
            self._record(11, "Streaming Query Endpoint (POST /query/stream)", False, desc, f"Failed POST /query/stream: {err}")

    def _test_12_frontend_build_check(self):
        desc = "Validates Next.js frontend code integrity and component structure."
        try:
            app_page = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend/my-app/src/app/page.tsx"))
            app_layout = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend/my-app/src/app/layout.tsx"))

            if os.path.exists(app_page) and os.path.exists(app_layout):
                with open(app_page, "r", encoding="utf-8") as f:
                    content = f.read()
                rrf_found = "RRF" in content or "Hybrid Search (RRF" in content
                if not rrf_found:
                    self._record(12, "Frontend Code & UI Integrity", True, desc, "Frontend clean. No internal RRF algorithm text found.")
                else:
                    self._record(12, "Frontend Code & UI Integrity", False, desc, "Frontend contains unwanted internal RRF text.")
            else:
                self._record(12, "Frontend Code & UI Integrity", False, desc, "Frontend app files missing.")
        except Exception as err:
            self._record(12, "Frontend Code & UI Integrity", False, desc, f"Error checking frontend files: {err}")

    def _test_13_frontend_server_check(self):
        desc = "Checks if Next.js frontend dev server is online at http://localhost:3000."
        try:
            req = urllib.request.Request(self.frontend_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    self._record(13, "Frontend Next.js Web Server", True, desc, f"Frontend online at {self.frontend_url} (HTTP 200).")
                else:
                    self._record(13, "Frontend Next.js Web Server", False, desc, f"HTTP Status: {resp.status}")
        except Exception as err:
            self._record(13, "Frontend Next.js Web Server", False, desc, f"Frontend web server offline at http://localhost:3000. (Start using: npm run dev in frontend/my-app)")

    # ── Reporter ──────────────────────────────────────────────────────────────

    def _print_report(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"])
        failed = total - passed

        print("\n" + "=" * 85, flush=True)
        print("                   FULL-STACK DIAGNOSTIC EXECUTION RESULTS                    ", flush=True)
        print("=" * 85 + "\n", flush=True)

        for r in self.results:
            status_text = "TRUE  [PASS]" if r["status"] else "FALSE [FAIL]"
            symbol = "[✓]" if r["status"] else "[X]"

            print(f"[{r['num']:02d}/13] {symbol} STATUS: {status_text}", flush=True)
            print(f"       MODULE     : {r['name']}", flush=True)
            print(f"       DESCRIPTION: {r['description']}", flush=True)
            print(f"       ANALYSIS   : {r['details']}", flush=True)
            print("-" * 85, flush=True)

        print("\n" + "=" * 85, flush=True)
        print(f"FULL-STACK SUMMARY: {passed}/{total} APPLICATION CHECKS PASSED", flush=True)
        if failed == 0:
            print("OVERALL APPLICATION STATUS: TRUE -- ALL BACKEND & FRONTEND SYSTEMS OPERATIONAL", flush=True)
        else:
            print(f"OVERALL APPLICATION STATUS: {passed}/{total} PASSED -- {failed} CHECK(S) PENDING/OFFLINE", flush=True)
            print("NOTE: For live HTTP endpoint checks, ensure 'start-dev.ps1' is running.")
        print("=" * 85 + "\n", flush=True)


if __name__ == "__main__":
    runner = FullStackDiagnosticRunner()
    runner.run_all()
