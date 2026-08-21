"""
clerk_verifier.py
─────────────────
Verifies Clerk-issued JWT tokens by fetching the JWKS (JSON Web Key Set)
from Clerk's well-known endpoint and decoding the token with RS256.

This is the "green sign from Clerk" — after this verification succeeds,
the backend trusts the identity and issues its own tokens.
"""

from typing import Dict, Any

import httpx
import jwt
from jwt import PyJWKClient

from app.core.config import settings


# Cache the JWKS client so we don't re-fetch keys on every request.
# PyJWKClient has built-in caching with a configurable lifespan.
_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    """
    Lazily initializes and returns the JWKS client.
    The JWKS URL is derived from the Clerk publishable key domain
    or set explicitly via CLERK_JWKS_URL env var.
    """
    global _jwks_client
    if _jwks_client is None:
        jwks_url = settings.CLERK_JWKS_URL
        if not jwks_url:
            raise ValueError(
                "CLERK_JWKS_URL is not configured. "
                "Set it to https://<your-clerk-domain>/.well-known/jwks.json"
            )
        _jwks_client = PyJWKClient(jwks_url, cache_keys=True, lifespan=3600)
    return _jwks_client


def verify_clerk_token(token: str) -> Dict[str, Any]:
    """
    Verifies a Clerk session JWT using the Clerk JWKS endpoint.

    Steps:
        1. Fetch the signing key from Clerk's JWKS endpoint
        2. Decode the JWT with RS256 verification
        3. Return the decoded payload

    Returns a dict with keys like:
        - sub       : Clerk user ID (e.g., "user_2xABC...")
        - email     : User's primary email (if present in token)
        - first_name: User's first name (if present)
        - last_name : User's last name (if present)
        - iat, exp  : Issued at / expiry timestamps

    Raises:
        jwt.InvalidTokenError  : If the token is invalid, expired, or tampered
        ValueError             : If JWKS URL is not configured
    """
    client = _get_jwks_client()
    signing_key = client.get_signing_key_from_jwt(token)

    decoded = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        options={
            # Clerk tokens may not always include an audience claim
            "verify_aud": False,
        },
    )
    return decoded


def get_clerk_user_info(clerk_token: str) -> Dict[str, Any]:
    """
    Convenience wrapper: verifies the Clerk token and also fetches
    full user details from the Clerk Backend API using the secret key.

    Returns user info including email, name, verification status.
    Falls back to token-only data if the API call fails.
    """
    # First verify the token
    token_data = verify_clerk_token(clerk_token)
    clerk_user_id = token_data.get("sub")

    if not clerk_user_id or not settings.CLERK_SECRET_KEY:
        return token_data

    # Fetch full user details from Clerk Backend API
    try:
        response = httpx.get(
            f"https://api.clerk.com/v1/users/{clerk_user_id}",
            headers={
                "Authorization": f"Bearer {settings.CLERK_SECRET_KEY}",
                "Content-Type": "application/json",
            },
            timeout=10.0,
        )
        response.raise_for_status()
        user_data = response.json()

        # Extract relevant fields
        primary_email = None
        for email_obj in user_data.get("email_addresses", []):
            if email_obj.get("id") == user_data.get("primary_email_address_id"):
                primary_email = email_obj.get("email_address")
                break

        return {
            "sub": clerk_user_id,
            "email": primary_email or token_data.get("email", ""),
            "first_name": user_data.get("first_name", ""),
            "last_name": user_data.get("last_name", ""),
            "image_url": user_data.get("image_url", ""),
            "has_verified_email": any(
                e.get("verification", {}).get("status") == "verified"
                for e in user_data.get("email_addresses", [])
            ),
        }

    except Exception as err:
        print(f"[ClerkVerifier] Warning: Could not fetch user details from Clerk API: {err}")
        # Fall back to token-only data
        return {
            "sub": clerk_user_id,
            "email": token_data.get("email", ""),
            "first_name": token_data.get("first_name", ""),
            "last_name": token_data.get("last_name", ""),
            "image_url": "",
            "has_verified_email": False,
        }
