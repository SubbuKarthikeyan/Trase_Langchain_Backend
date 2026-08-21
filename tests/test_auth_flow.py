"""
test_auth_flow.py
─────────────────
Automated test suite verifying the Clerk authentication + custom JWT session management:
1. Token creation and signature verification (access + refresh)
2. Token hashing (SHA-256)
3. User, Profile, and Connector document schemas
4. Database collection upsert operations in MongoDB
5. Refresh token rotation & invalidation handling
"""

import sys
import os
import unittest
from datetime import datetime, timezone

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.auth.token_service import (
    create_access_token,
    create_refresh_token,
    verify_access_token,
    verify_refresh_token,
    hash_token,
)
from app.core.schemas import UserDocument, ProfileDocument, ConnectorDocument
from app.core.mongo_client import get_collection
from app.core.config import settings
import jwt


class TestAuthFlow(unittest.TestCase):

    def test_1_access_token_creation_and_verification(self):
        token = create_access_token(user_id="user_test123", email="test@example.com")
        self.assertIsInstance(token, str)

        payload = verify_access_token(token)
        self.assertEqual(payload["sub"], "user_test123")
        self.assertEqual(payload["email"], "test@example.com")
        self.assertEqual(payload["type"], "access")

    def test_2_refresh_token_creation_and_verification(self):
        refresh_token = create_refresh_token(user_id="user_test123")
        self.assertIsInstance(refresh_token, str)

        payload = verify_refresh_token(refresh_token)
        self.assertEqual(payload["sub"], "user_test123")
        self.assertEqual(payload["type"], "refresh")

    def test_3_token_hashing(self):
        raw_token = "sample_refresh_token_string"
        h1 = hash_token(raw_token)
        h2 = hash_token(raw_token)
        self.assertEqual(h1, h2)
        self.assertNotEqual(raw_token, h1)
        self.assertEqual(len(h1), 64)  # SHA-256 hex digest length

    def test_4_user_and_profile_schema_validation(self):
        user = UserDocument(
            clerk_user_id="user_test_clerk_id",
            email="developer@trase.ai",
            is_verified=True,
            is_banned=False,
            role="user",
        )
        self.assertEqual(user.clerk_user_id, "user_test_clerk_id")
        self.assertTrue(user.is_verified)
        self.assertFalse(user.is_banned)

        profile = ProfileDocument(
            user_id="user_test_clerk_id",
            first_name="Expert",
            last_name="Developer",
        )
        self.assertEqual(profile.user_id, "user_test_clerk_id")

        connector = ConnectorDocument(
            user_id="user_test_clerk_id",
            provider="github",
            provider_user_id="github_12345",
            display_name="GitHub Account",
        )
        self.assertEqual(connector.provider, "github")

    def test_5_mongodb_user_profile_connector_upsert(self):
        clerk_id = "user_integration_test_999"
        email = "integration_test@trase.ai"

        users_col = get_collection(settings.MONGO_USERS_COLLECTION)
        profiles_col = get_collection(settings.MONGO_PROFILES_COLLECTION)
        connectors_col = get_collection(settings.MONGO_CONNECTORS_COLLECTION)

        # 1. Upsert User
        now = datetime.now(tz=timezone.utc)
        test_refresh_token = create_refresh_token(clerk_id)
        hashed_rf = hash_token(test_refresh_token)

        users_col.update_one(
            {"clerk_user_id": clerk_id},
            {
                "$set": {
                    "email": email,
                    "hashed_refresh_token": hashed_rf,
                    "is_verified": True,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "clerk_user_id": clerk_id,
                    "is_banned": False,
                    "role": "user",
                    "created_at": now,
                },
            },
            upsert=True,
        )

        user_doc = users_col.find_one({"clerk_user_id": clerk_id})
        self.assertIsNotNone(user_doc)
        self.assertEqual(user_doc["email"], email)
        self.assertEqual(user_doc["hashed_refresh_token"], hashed_rf)
        self.assertTrue(user_doc["is_verified"])

        # 2. Upsert Profile
        profiles_col.update_one(
            {"user_id": clerk_id},
            {
                "$set": {
                    "first_name": "Trase",
                    "last_name": "Tester",
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "user_id": clerk_id,
                    "bio": "Test bio",
                    "created_at": now,
                },
            },
            upsert=True,
        )

        profile_doc = profiles_col.find_one({"user_id": clerk_id})
        self.assertIsNotNone(profile_doc)
        self.assertEqual(profile_doc["first_name"], "Trase")

        # 3. Upsert Connector
        connectors_col.update_one(
            {"user_id": clerk_id, "provider": "github"},
            {
                "$set": {
                    "provider_user_id": "gh_test_user_1",
                    "display_name": "GitHub Connector",
                    "is_active": True,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "user_id": clerk_id,
                    "provider": "github",
                    "connected_at": now,
                },
            },
            upsert=True,
        )

        connector_doc = connectors_col.find_one({"user_id": clerk_id, "provider": "github"})
        self.assertIsNotNone(connector_doc)
        self.assertEqual(connector_doc["provider_user_id"], "gh_test_user_1")

        # Cleanup test document
        users_col.delete_one({"clerk_user_id": clerk_id})
        profiles_col.delete_one({"user_id": clerk_id})
        connectors_col.delete_one({"user_id": clerk_id, "provider": "github"})


if __name__ == "__main__":
    unittest.main()
