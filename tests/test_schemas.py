"""
test_schemas.py
───────────────
Unit tests validating Pydantic database models and schema definitions.
"""

import unittest
from datetime import datetime, timezone
from app.core.schemas import (
    FileRegistryEntry,
    VectorChunkMetadata,
    ChatMessage,
    ChatSession,
    BusStop,
    OperatingHours,
    FareStructure,
    BusRoute,
    Role,
    QueryMode
)
from app.core.config import settings


class TestDatabaseSchemas(unittest.TestCase):

    def test_file_registry_schema(self):
        entry = FileRegistryEntry(
            filename="bus_data_modified.csv",
            file_hash="d41d8cd98f00b204e9800998ecf8427e",
            chunk_count=42
        )
        self.assertEqual(entry.filename, "bus_data_modified.csv")
        self.assertEqual(entry.chunk_count, 42)
        self.assertIsInstance(entry.ingested_at, datetime)

    def test_vector_chunk_metadata_schema(self):
        metadata = VectorChunkMetadata(
            source="bus_data_modified.csv",
            file_hash="d41d8cd98f00b204e9800998ecf8427e",
            chunk_id=1
        )
        self.assertEqual(metadata.source, "bus_data_modified.csv")
        self.assertEqual(metadata.chunk_id, 1)

    def test_chat_session_schema(self):
        msg = ChatMessage(
            message_id="msg-123",
            role=Role.USER,
            content="What time does bus 21B run?",
            mode=QueryMode.RAG
        )
        session = ChatSession(
            session_id="sess-456",
            messages=[msg]
        )
        self.assertEqual(session.session_id, "sess-456")
        self.assertEqual(len(session.messages), 1)
        self.assertEqual(session.messages[0].content, "What time does bus 21B run?")

    def test_bus_route_schema(self):
        stop1 = BusStop(stop_name="Terminal A", sequence_order=1, fare_from_origin=0.0)
        stop2 = BusStop(stop_name="Station B", sequence_order=2, fare_from_origin=15.0)
        op_hours = OperatingHours(first_bus="06:00", last_bus="22:00", frequency_mins=15)
        fare_struct = FareStructure(base_fare=10.0, per_km_rate=2.5)

        route = BusRoute(
            route_id="R-101",
            route_number="21B",
            origin="Terminal A",
            destination="Station B",
            stops=[stop1, stop2],
            operating_hours=op_hours,
            fare_structure=fare_struct
        )
        self.assertEqual(route.route_id, "R-101")
        self.assertEqual(len(route.stops), 2)
        self.assertEqual(route.stops[1].fare_from_origin, 15.0)

    def test_config_collections(self):
        self.assertEqual(settings.MONGO_REGISTRY_COLLECTION, "file_registry")
        self.assertEqual(settings.MONGO_SESSION_COLLECTION, "chat_sessions")
        self.assertEqual(settings.MONGO_STRUCTURED_ROUTES_COLLECTION, "structured_bus_routes")


if __name__ == "__main__":
    unittest.main()
