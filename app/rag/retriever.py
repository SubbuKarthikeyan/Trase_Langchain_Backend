"""
retriever.py
─────────────
Production MongoDB RAG Retriever for Trase Bus Travel API.
Retrieves relevant bus documents directly from MongoDB using Atlas Vector Search,
MongoDB vector cosine scan, or MongoDB document regex search.

Zero local disk file fallback — strict database operation.
"""

import math
from pymongo import MongoClient
from app.rag.lc_vectorstore import get_vectorstore
from app.core.config import settings


def _cosine_similarity(v1: list, v2: list) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2) if n1 and n2 else 0.0


def _mongodb_cosine_search(query_embedding: list, top_k: int) -> list[str]:
    """Cosine vector search directly over stored MongoDB collection documents."""
    client = MongoClient(settings.MONGO_URL)
    col = client[settings.MONGO_DB_NAME][settings.MONGO_COLLECTION_NAME]
    cursor = col.find({}, {"page_content": 1, "embedding": 1})
    scored = []
    for doc in cursor:
        emb = doc.get("embedding", [])
        if emb and len(emb) == len(query_embedding):
            scored.append((_cosine_similarity(query_embedding, emb), doc.get("page_content", "")))
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        print(f"  [MongoDB Cosine Search] top scores: {[round(s, 4) for s, _ in scored[:top_k]]}")
    return [text for _, text in scored[:top_k]]


def _mongodb_structured_search(question: str, top_k: int) -> list[str]:
    """
    Directly queries structured_bus_routes in MongoDB by extracted origin, destination,
    or key travel terms to guarantee accurate bus route retrieval.
    """
    client = MongoClient(settings.MONGO_URL)
    db = client[settings.MONGO_DB_NAME]
    col = db[settings.MONGO_STRUCTURED_ROUTES_COLLECTION]

    # Clean words to detect locations
    stop_words = {
        "what", "when", "where", "which", "how", "who", "whom", "this", "that",
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "can", "could", "should", "would",
        "bus", "buses", "route", "routes", "travel", "travelling", "ticket", "tickets",
        "fare", "fares", "price", "cost", "timing", "timings", "time", "available",
        "availability", "from", "to", "for", "in", "at", "by", "with", "any", "some",
        "please", "tell", "show", "give", "find", "search", "me", "i", "we", "my",
    }
    tokens = [w.strip("?,.!;:") for w in question.lower().split()]
    meaningful = [t for t in tokens if len(t) > 2 and t not in stop_words]

    if not meaningful:
        return []

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
        print(f"  [MongoDB Structured Search] Retrieved {len(results)} matching route(s).")
    return results


def _mongodb_regex_search(question: str, top_k: int) -> list[str]:
    """Text regex search directly over MongoDB collections when vector search is unavailable."""
    client = MongoClient(settings.MONGO_URL)
    db = client[settings.MONGO_DB_NAME]

    stop_words = {
        "what", "when", "where", "which", "how", "who", "the", "a", "an", "is", "are",
        "was", "were", "bus", "buses", "ticket", "tickets", "fare", "fares", "price",
        "cost", "time", "available", "from", "to", "for", "in", "at", "any", "some",
        "please", "tell", "show", "give", "find", "search", "me", "i",
    }
    keywords = [w.strip("?,.!;:") for w in question.lower().split() if len(w) > 2 and w.lower() not in stop_words]
    if not keywords:
        keywords = [w for w in question.lower().split() if len(w) > 2]
    if not keywords:
        return []

    regex_pattern = "|".join(keywords)
    col = db[settings.MONGO_COLLECTION_NAME]
    cursor = col.find({"page_content": {"$regex": regex_pattern, "$options": "i"}}, {"page_content": 1}).limit(top_k)

    results = [doc.get("page_content", "") for doc in cursor if doc.get("page_content")]
    if results:
        print(f"  [MongoDB Regex Search] Retrieved {len(results)} matching document(s) from MongoDB.")
    return results


class Retriever:
    """
    Production RAG Retriever — retrieves document context strictly from MongoDB.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def retrieve(self, question: str, top_k: int = None) -> list[str]:
        k = top_k or self.top_k
        print("\n[Retriever] Question:", question)

        if not settings.MONGO_URL:
            print("  [Retriever Error] MONGO_URL is not configured.")
            return []

        # Tier 1: MongoDB Atlas Vector Search
        try:
            vectorstore = get_vectorstore()
            lc_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k},
            )
            docs = lc_retriever.invoke(question)
            if docs:
                print(f"  [MongoDB Atlas Vector Search] Retrieved {len(docs)} document chunks.")
                return [doc.page_content for doc in docs]
        except Exception as err:
            print(f"  [MongoDB Atlas Search Warning] {err}")

        # Tier 2: Direct MongoDB Cosine Vector Scan
        try:
            vectorstore = get_vectorstore()
            query_embedding = vectorstore._embedding.embed_query(question)
            results = _mongodb_cosine_search(query_embedding, k)
            if results:
                return results
        except Exception as err:
            print(f"  [MongoDB Cosine Search Warning] {err}")

        # Tier 3: MongoDB Structured Routes Search
        try:
            results = _mongodb_structured_search(question, k)
            if results:
                return results
        except Exception as err:
            print(f"  [MongoDB Structured Search Warning] {err}")

        # Tier 4: Direct MongoDB Regex Text Search
        try:
            return _mongodb_regex_search(question, k)
        except Exception as err:
            print(f"  [MongoDB Regex Search Warning] {err}")
            return []
