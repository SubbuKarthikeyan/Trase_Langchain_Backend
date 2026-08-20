"""
retriever.py
─────────────
Production MongoDB RAG Retriever for Trase Bus Travel API.

Architecture:
    Hybrid Search with Reciprocal Rank Fusion (RRF).

    Three retrieval legs run in parallel:
        Leg 1 — Semantic: Atlas Vector Search (with cosine scan fallback)
        Leg 2 — Keyword:  MongoDB regex text search
        Leg 3 — Structured: Exact route field matching

    Results from all legs are fused using the RRF formula:
        RRF_score(doc) = Σ  1 / (k + rank_i(doc))

    When HYBRID_SEARCH_ENABLED=false, falls back to the legacy waterfall
    retriever for backward compatibility.

Zero local disk file fallback — strict database operation.
"""

import math
import hashlib
from pymongo import MongoClient

from app.rag.lc_vectorstore import get_vectorstore
from app.core.config import settings
from app.core.mongo_client import get_collection


# ──────────────────────────────────────────────────────────────────────────────
# RRF Fusion Engine
# ──────────────────────────────────────────────────────────────────────────────

def _content_key(text: str) -> str:
    """
    Generates a deduplication key for a chunk of text.
    Uses the first 200 chars (normalized) to detect near-duplicate content
    returned by different retrieval legs (e.g., vector vs structured may
    return the same bus route in slightly different formats).
    """
    normalized = text.strip().lower()[:200]
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """
    Reciprocal Rank Fusion across multiple ranked retrieval lists.

    Each document receives a score:  Σ 1/(k + rank)  across all lists
    it appears in. Documents appearing in multiple legs get a natural boost.

    Args:
        ranked_lists: List of ranked result lists (each list is ordered
                      by relevance, best first).
        k:           Smoothing constant. Standard value is 60.
                     Lower k → top ranks get heavier weight.
                     Higher k → more uniform weighting.

    Returns:
        Fused list of document texts, sorted by RRF score descending.
    """
    scores: dict[str, float] = {}      # content_key → cumulative RRF score
    key_to_text: dict[str, str] = {}   # content_key → original text (keep first seen)

    for leg_idx, ranked_list in enumerate(ranked_lists):
        for rank, doc_text in enumerate(ranked_list, start=1):
            key = _content_key(doc_text)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in key_to_text:
                key_to_text[key] = doc_text

    # Sort by fused score descending
    sorted_keys = sorted(scores.keys(), key=lambda k_: scores[k_], reverse=True)
    return [key_to_text[k_] for k_ in sorted_keys]


def _rrf_fuse_with_scores(ranked_lists: list[list[str]], k: int = 60) -> list[dict]:
    """
    Same as _rrf_fuse but returns detailed score breakdowns for debugging.

    Returns:
        List of dicts: [{"text": str, "rrf_score": float, "appeared_in_legs": list[int]}, ...]
    """
    scores: dict[str, float] = {}
    key_to_text: dict[str, str] = {}
    key_to_legs: dict[str, list[int]] = {}

    for leg_idx, ranked_list in enumerate(ranked_lists):
        for rank, doc_text in enumerate(ranked_list, start=1):
            key = _content_key(doc_text)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in key_to_text:
                key_to_text[key] = doc_text
            if key not in key_to_legs:
                key_to_legs[key] = []
            key_to_legs[key].append(leg_idx)

    sorted_keys = sorted(scores.keys(), key=lambda k_: scores[k_], reverse=True)
    return [
        {
            "text": key_to_text[k_],
            "rrf_score": round(scores[k_], 6),
            "appeared_in_legs": key_to_legs[k_],
        }
        for k_ in sorted_keys
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval Leg 1: Semantic Search (Atlas Vector + Cosine Fallback)
# ──────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(v1: list, v2: list) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2) if n1 and n2 else 0.0


def _semantic_search(question: str, top_k: int) -> list[str]:
    """
    Semantic retrieval leg.
    Primary: MongoDB Atlas Vector Search (fast, index-based).
    Fallback: Manual cosine scan over stored embeddings.
    """
    # Try Atlas Vector Search first
    try:
        vectorstore = get_vectorstore()
        lc_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )
        docs = lc_retriever.invoke(question)
        if docs:
            print(f"  [Leg 1 - Semantic] Atlas Vector Search -> {len(docs)} chunk(s)")
            return [doc.page_content for doc in docs]
    except Exception as err:
        print(f"  [Leg 1 - Semantic] Atlas Vector Search failed: {err}")

    # Fallback: manual cosine scan
    try:
        vectorstore = get_vectorstore()
        query_embedding = vectorstore._embedding.embed_query(question)
        col = get_collection(settings.MONGO_COLLECTION_NAME)
        cursor = col.find({}, {"page_content": 1, "embedding": 1})

        scored = []
        for doc in cursor:
            emb = doc.get("embedding", [])
            if emb and len(emb) == len(query_embedding):
                scored.append((_cosine_similarity(query_embedding, emb), doc.get("page_content", "")))
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored:
            results = [text for _, text in scored[:top_k]]
            print(f"  [Leg 1 - Semantic] Cosine scan -> {len(results)} chunk(s), "
                  f"top scores: {[round(s, 4) for s, _ in scored[:top_k]]}")
            return results
    except Exception as err:
        print(f"  [Leg 1 - Semantic] Cosine scan failed: {err}")

    return []


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval Leg 2: Keyword Search (MongoDB Regex)
# ──────────────────────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "what", "when", "where", "which", "how", "who", "the", "a", "an", "is", "are",
    "was", "were", "bus", "buses", "ticket", "tickets", "fare", "fares", "price",
    "cost", "time", "available", "from", "to", "for", "in", "at", "any", "some",
    "please", "tell", "show", "give", "find", "search", "me", "i",
}


def _extract_keywords(question: str) -> list[str]:
    """Extracts meaningful keywords from a user question."""
    tokens = [w.strip("?,.!;:") for w in question.lower().split()]
    keywords = [t for t in tokens if len(t) > 2 and t not in _STOP_WORDS]
    if not keywords:
        keywords = [w for w in question.lower().split() if len(w) > 2]
    return keywords


def _keyword_search(question: str, top_k: int) -> list[str]:
    """
    Keyword retrieval leg.
    Uses MongoDB regex matching on page_content in the vector store collection.
    """
    keywords = _extract_keywords(question)
    if not keywords:
        return []

    try:
        regex_pattern = "|".join(keywords)
        col = get_collection(settings.MONGO_COLLECTION_NAME)
        cursor = col.find(
            {"page_content": {"$regex": regex_pattern, "$options": "i"}},
            {"page_content": 1},
        ).limit(top_k)

        results = [doc.get("page_content", "") for doc in cursor if doc.get("page_content")]
        if results:
            print(f"  [Leg 2 - Keyword] Regex search -> {len(results)} chunk(s) "
                  f"(keywords: {keywords[:5]})")
        return results
    except Exception as err:
        print(f"  [Leg 2 - Keyword] Regex search failed: {err}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval Leg 3: Structured Route Search
# ──────────────────────────────────────────────────────────────────────────────

_STRUCTURED_STOP_WORDS = {
    "what", "when", "where", "which", "how", "who", "whom", "this", "that",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "can", "could", "should", "would",
    "bus", "buses", "route", "routes", "travel", "travelling", "ticket", "tickets",
    "fare", "fares", "price", "cost", "timing", "timings", "time", "available",
    "availability", "from", "to", "for", "in", "at", "by", "with", "any", "some",
    "please", "tell", "show", "give", "find", "search", "me", "i", "we", "my",
}


def _structured_search(question: str, top_k: int) -> list[str]:
    """
    Structured retrieval leg.
    Queries structured_bus_routes collection by extracted location/bus terms
    to guarantee accurate bus route retrieval.
    """
    tokens = [w.strip("?,.!;:") for w in question.lower().split()]
    meaningful = [t for t in tokens if len(t) > 2 and t not in _STRUCTURED_STOP_WORDS]

    if not meaningful:
        return []

    try:
        col = get_collection(settings.MONGO_STRUCTURED_ROUTES_COLLECTION)

        # Build regex query across origin, destination, bus_name, stops, bus_type
        or_clauses = []
        for term in meaningful:
            or_clauses.extend([
                {"origin": {"$regex": term, "$options": "i"}},
                {"destination": {"$regex": term, "$options": "i"}},
                {"bus_name": {"$regex": term, "$options": "i"}},
                {"bus_type": {"$regex": term, "$options": "i"}},
                {"stops": {"$regex": term, "$options": "i"}},
            ])

        cursor = col.find({"$or": or_clauses}).limit(top_k)
        results = []
        for doc in cursor:
            chunk = (
                f"Bus: {doc.get('bus_name', 'Express')} ({doc.get('bus_type', 'Standard')})\n"
                f"Route: {doc.get('origin', '')} to {doc.get('destination', '')}\n"
                f"Departure: {doc.get('departure_time', 'N/A')} | Arrival: {doc.get('arrival_time', 'N/A')} | Duration: {doc.get('travel_time', 'N/A')}\n"
                f"Seats: {doc.get('available_seats', 'N/A')} ({doc.get('seat_type', 'Seat')})\n"
                f"Base Minimum Fare: Rs. {doc.get('min_fare', 'N/A')} | Additional Stop Fare: Rs. {doc.get('additional_stop_fare', 'N/A')}\n"
                f"Facilities: {doc.get('facilities', 'N/A')}\n"
                f"Route Stops: {doc.get('stops', 'N/A')}"
            )
            results.append(chunk)

        if results:
            print(f"  [Leg 3 - Structured] Route search -> {len(results)} route(s)")
        return results
    except Exception as err:
        print(f"  [Leg 3 - Structured] Route search failed: {err}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Legacy Waterfall Retriever (preserved for HYBRID_SEARCH_ENABLED=false)
# ──────────────────────────────────────────────────────────────────────────────

def _legacy_retrieve(question: str, top_k: int) -> list[str]:
    """
    Original waterfall retrieval — tries each tier sequentially,
    returns the first tier that produces results.
    """
    # Tier 1: Atlas Vector Search
    results = _semantic_search(question, top_k)
    if results:
        return results

    # Tier 2: Structured Routes
    results = _structured_search(question, top_k)
    if results:
        return results

    # Tier 3: Keyword/Regex
    return _keyword_search(question, top_k)


# ──────────────────────────────────────────────────────────────────────────────
# Public Retriever Class
# ──────────────────────────────────────────────────────────────────────────────

class Retriever:
    """
    Production RAG Retriever — retrieves document context strictly from MongoDB.

    Modes:
        hybrid=True  (default): Parallel retrieval + RRF fusion.
        hybrid=False           : Legacy waterfall (Tier 1 -> 2 -> 3).
    """

    def __init__(self, top_k: int = 5, rrf_k: int = None, hybrid: bool = None):
        self.top_k = top_k
        self.rrf_k = rrf_k if rrf_k is not None else settings.RRF_K
        self.hybrid = hybrid if hybrid is not None else settings.HYBRID_SEARCH_ENABLED

    def retrieve(self, question: str, top_k: int = None) -> list[str]:
        k = top_k or self.top_k
        print(f"\n[Retriever] Question: {question}")
        print(f"[Retriever] Mode: {'hybrid (RRF k={})'.format(self.rrf_k) if self.hybrid else 'legacy waterfall'}")

        if not settings.MONGO_URL:
            print("  [Retriever Error] MONGO_URL is not configured.")
            return []

        if not self.hybrid:
            return _legacy_retrieve(question, k)

        return self._hybrid_retrieve(question, k)

    def _hybrid_retrieve(self, question: str, top_k: int) -> list[str]:
        """
        Runs all three retrieval legs in parallel (sequentially in Python,
        but each leg is independent), then fuses with RRF.
        """
        leg_names = ["Semantic", "Keyword", "Structured"]
        ranked_lists: list[list[str]] = []

        # Leg 1: Semantic Search
        try:
            semantic_results = _semantic_search(question, top_k)
            ranked_lists.append(semantic_results)
        except Exception as err:
            print(f"  [Hybrid] Semantic leg exception: {err}")
            ranked_lists.append([])

        # Leg 2: Keyword Search
        try:
            keyword_results = _keyword_search(question, top_k)
            ranked_lists.append(keyword_results)
        except Exception as err:
            print(f"  [Hybrid] Keyword leg exception: {err}")
            ranked_lists.append([])

        # Leg 3: Structured Search
        try:
            structured_results = _structured_search(question, top_k)
            ranked_lists.append(structured_results)
        except Exception as err:
            print(f"  [Hybrid] Structured leg exception: {err}")
            ranked_lists.append([])

        # Report per-leg counts
        for i, (name, rlist) in enumerate(zip(leg_names, ranked_lists)):
            print(f"  [Hybrid] {name}: {len(rlist)} result(s)")

        # Filter out empty legs before fusion
        non_empty = [rl for rl in ranked_lists if rl]
        if not non_empty:
            print("  [Hybrid] All legs returned empty. No results.")
            return []

        if len(non_empty) == 1:
            # Only one leg produced results — no fusion needed
            print("  [Hybrid] Single leg produced results, skipping RRF fusion.")
            return non_empty[0][:top_k]

        # RRF Fusion
        fused = _rrf_fuse(non_empty, k=self.rrf_k)
        print(f"  [Hybrid] RRF fused {sum(len(r) for r in non_empty)} candidates -> {len(fused)} unique, returning top {top_k}")

        return fused[:top_k]


    def retrieve_with_debug(self, question: str, top_k: int = None) -> dict:
        """
        Retrieves results with full debug information for the /query/debug endpoint.
        Returns detailed per-leg rankings and RRF scores.
        """
        k = top_k or self.top_k

        if not settings.MONGO_URL:
            return {"error": "MONGO_URL is not configured", "results": []}

        leg_results = {}

        # Leg 1: Semantic
        try:
            leg_results["semantic"] = _semantic_search(question, k)
        except Exception as err:
            leg_results["semantic"] = []
            leg_results["semantic_error"] = str(err)

        # Leg 2: Keyword
        try:
            leg_results["keyword"] = _keyword_search(question, k)
        except Exception as err:
            leg_results["keyword"] = []
            leg_results["keyword_error"] = str(err)

        # Leg 3: Structured
        try:
            leg_results["structured"] = _structured_search(question, k)
        except Exception as err:
            leg_results["structured"] = []
            leg_results["structured_error"] = str(err)

        # RRF Fusion with scores
        non_empty = [
            leg_results[leg] for leg in ["semantic", "keyword", "structured"]
            if leg_results.get(leg)
        ]

        if non_empty:
            fused_detail = _rrf_fuse_with_scores(non_empty, k=self.rrf_k)
        else:
            fused_detail = []

        return {
            "question": question,
            "mode": "hybrid" if self.hybrid else "legacy",
            "rrf_k": self.rrf_k,
            "top_k": k,
            "legs": {
                "semantic": {
                    "count": len(leg_results.get("semantic", [])),
                    "results": [text[:150] + "..." if len(text) > 150 else text
                                for text in leg_results.get("semantic", [])],
                    "error": leg_results.get("semantic_error"),
                },
                "keyword": {
                    "count": len(leg_results.get("keyword", [])),
                    "results": [text[:150] + "..." if len(text) > 150 else text
                                for text in leg_results.get("keyword", [])],
                    "error": leg_results.get("keyword_error"),
                },
                "structured": {
                    "count": len(leg_results.get("structured", [])),
                    "results": [text[:150] + "..." if len(text) > 150 else text
                                for text in leg_results.get("structured", [])],
                    "error": leg_results.get("structured_error"),
                },
            },
            "fused": fused_detail[:k],
            "fused_count": len(fused_detail),
        }
