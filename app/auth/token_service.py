"""
token_service.py
────────────────
JWT token creation and verification for Trase session management.

- Access tokens  : short-lived (30 min), sent in Authorization header
- Refresh tokens : long-lived (7 days), stored as HttpOnly cookie
- Refresh tokens are SHA-256 hashed before storing in MongoDB
"""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import jwt

from app.core.config import settings


# ── Token Creation ───────────────────────────────────────────────────────────

def create_access_token(
    user_id: str,
    email: str,
    role: str = "user",
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Creates a short-lived JWT access token.

    Payload includes:
        sub  : clerk_user_id
        email: user email
        role : user role
        type : "access"
        iat  : issued at
        exp  : expiry (30 minutes from now)
    """
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """
    Creates a long-lived JWT refresh token.

    Payload includes:
        sub  : clerk_user_id
        type : "refresh"
        iat  : issued at
        exp  : expiry (7 days from now)
    """
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": user_id,
        "type": "refresh",
        "iat": now,
        "exp": now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    }
    return jwt.encode(payload, settings.JWT_REFRESH_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


# ── Token Verification ──────────────────────────────────────────────────────

def verify_access_token(token: str) -> Dict[str, Any]:
    """
    Decodes and validates an access token.
    Raises jwt.InvalidTokenError on failure (expired, tampered, wrong type).
    """
    payload = jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
    )
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("Token is not an access token")
    return payload


def verify_refresh_token(token: str) -> Dict[str, Any]:
    """
    Decodes and validates a refresh token.
    Raises jwt.InvalidTokenError on failure.
    """
    payload = jwt.decode(
        token,
        settings.JWT_REFRESH_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
    )
    if payload.get("type") != "refresh":
        raise jwt.InvalidTokenError("Token is not a refresh token")
    return payload


# ── Hashing ──────────────────────────────────────────────────────────────────

def hash_token(token: str) -> str:
    """
    Returns the SHA-256 hex digest of a token.
    Used to store refresh tokens securely in MongoDB — we never persist raw tokens.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
