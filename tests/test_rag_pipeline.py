"""
test_rag_pipeline.py
────────────────────
Unit test suite verifying the production MongoDB RAG pipeline, loader,
and retriever configuration.
"""

import unittest
from app.rag.retriever import Retriever
from app.rag.loader import load_document, load_structured_routes
from app.core.config import settings


class TestRAGPipeline(unittest.TestCase):

    def test_loader_structured_routes(self):
        records = load_structured_routes("data/raw/bus_data_modified.csv")
        self.assertGreater(len(records), 0)
        self.assertIn("route_id", records[0])
        self.assertIn("origin", records[0])
        self.assertIn("destination", records[0])

    def test_retriever_initialization(self):
        retriever = Retriever(top_k=3)
        self.assertEqual(retriever.top_k, 3)

    def test_config_db_settings(self):
        self.assertEqual(settings.MONGO_DB_NAME, "trase_bus_db")
        self.assertEqual(settings.MONGO_COLLECTION_NAME, "bus_routes_v2")


if __name__ == "__main__":
    unittest.main()
