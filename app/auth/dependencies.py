"""
dependencies.py
───────────────
FastAPI dependency injection for authentication.

Usage in route handlers:
    from app.auth.dependencies import get_current_user, get_optional_user

    @router.get("/protected")
    def protected_route(user: dict = Depends(get_current_user)):
        return {"message": f"Hello, {user['email']}"}

    @router.get("/mixed")
    def mixed_route(user: dict | None = Depends(get_optional_user)):
        if user:
            return {"message": f"Hello, {user['email']}"}
        return {"message": "Hello, guest"}
"""

from typing import Optional, Dict, Any

from fastapi import Request, HTTPException, status

from app.auth.token_service import verify_access_token
from app.core.mongo_client import get_collection
from app.core.config import settings

import jwt


def get_current_user(request: Request) -> Dict[str, Any]:
    """
    Extracts and validates the access token from the Authorization header.

    Expected header format:
        Authorization: Bearer <access_token>

    Returns the decoded JWT payload if valid.
    Raises HTTP 401 if missing, malformed, expired, or user is banned.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header.split(" ", 1)[1]

    try:
        payload = verify_access_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid access token: {err}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is banned in the database
    user_id = payload.get("sub")
    if user_id:
        users_col = get_collection(settings.MONGO_USERS_COLLECTION)
        user_doc = users_col.find_one(
            {"clerk_user_id": user_id},
            {"is_banned": 1},
        )
        if user_doc and user_doc.get("is_banned"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account has been suspended",
            )

    return payload


def get_optional_user(request: Request) -> Optional[Dict[str, Any]]:
    """
    Same as get_current_user but returns None instead of raising 401
    when no token is present. Useful for endpoints that work for both
    authenticated and unauthenticated users.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header.split(" ", 1)[1]

    try:
        payload = verify_access_token(token)
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

    return payload
