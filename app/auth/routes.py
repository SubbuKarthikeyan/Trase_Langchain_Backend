"""
routes.py
─────────
Auth API endpoints for Trase Bus Travel API.

Flow:
    1. Frontend signs in via Clerk → gets Clerk session token
    2. Frontend sends Clerk token to POST /api/auth/session
    3. Backend verifies Clerk token (the "green sign")
    4. Backend upserts user + profile in MongoDB
    5. Backend generates its own access + refresh tokens
    6. Access token → JSON response body
    7. Refresh token → HttpOnly cookie (secure, SameSite=Lax)
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Request, Response, HTTPException, status, Depends
from pydantic import BaseModel

import jwt

from app.auth.clerk_verifier import verify_clerk_token, get_clerk_user_info
from app.auth.token_service import (
    create_access_token,
    create_refresh_token,
    verify_refresh_token,
    hash_token,
)
from app.auth.dependencies import get_current_user
from app.core.mongo_client import get_collection
from app.core.config import settings


router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# ── Request / Response Schemas ───────────────────────────────────────────────

class SessionRequest(BaseModel):
    """Body for the token exchange endpoint."""
    clerk_token: str


class SessionResponse(BaseModel):
    """Response with access token and user info."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: dict


# ── Helper: Set Refresh Token Cookie ─────────────────────────────────────────

def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    """
    Sets the refresh token as an HttpOnly cookie.

    Security settings:
        - httponly : prevents JavaScript access (XSS protection)
        - secure   : only sent over HTTPS (disabled in dev for localhost)
        - samesite : Lax (sent on top-level navigations, not cross-site)
        - max_age  : matches refresh token expiry
        - path     : restricted to auth refresh endpoint
    """
    max_age_seconds = settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
    response.set_cookie(
        key="trase_refresh_token",
        value=refresh_token,
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax",
        max_age=max_age_seconds,
        path="/api/auth",  # Only sent to auth endpoints
    )


def _clear_refresh_cookie(response: Response) -> None:
    """Removes the refresh token cookie."""
    response.delete_cookie(
        key="trase_refresh_token",
        path="/api/auth",
    )


# ── POST /api/auth/session ───────────────────────────────────────────────────

@router.post("/session", response_model=SessionResponse)
def create_session(body: SessionRequest, response: Response):
    """
    Exchange a Clerk session token for Trase access + refresh tokens.

    This is the main authentication endpoint. The frontend calls this
    after a successful Clerk sign-in, sending the Clerk token. The backend:
        1. Verifies the Clerk token via JWKS (the "green sign")
        2. Fetches user details from Clerk Backend API
        3. Upserts user document in MongoDB (with is_verified, is_banned)
        4. Upserts profile document in MongoDB
        5. Generates access token (30 min) + refresh token (7 days)
        6. Stores hashed refresh token in the user document
        7. Returns access token in body, refresh token as HttpOnly cookie
    """
    # Step 1: Verify the Clerk token
    try:
        clerk_user_info = get_clerk_user_info(body.clerk_token)
    except jwt.InvalidTokenError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Clerk token: {err}",
        )
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(err),
        )

    clerk_user_id = clerk_user_info["sub"]
    email = clerk_user_info.get("email", "")
    first_name = clerk_user_info.get("first_name", "")
    last_name = clerk_user_info.get("last_name", "")
    image_url = clerk_user_info.get("image_url", "")
    has_verified_email = clerk_user_info.get("has_verified_email", False)

    now = datetime.now(tz=timezone.utc)

    # Step 2: Check if user is banned
    users_col = get_collection(settings.MONGO_USERS_COLLECTION)
    existing_user = users_col.find_one({"clerk_user_id": clerk_user_id})

    if existing_user and existing_user.get("is_banned"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been suspended. Contact support.",
        )

    # Step 3: Generate our own tokens
    access_token = create_access_token(clerk_user_id, email)
    refresh_token = create_refresh_token(clerk_user_id)
    hashed_refresh = hash_token(refresh_token)

    # Step 4: Upsert user document
    users_col.update_one(
        {"clerk_user_id": clerk_user_id},
        {
            "$set": {
                "email": email,
                "hashed_refresh_token": hashed_refresh,
                "is_verified": has_verified_email,
                "updated_at": now,
            },
            "$setOnInsert": {
                "clerk_user_id": clerk_user_id,
                "is_banned": False,
                "role": "user",
                "created_at": now,
            },
        },
        upsert=True,
    )

    # Step 5: Upsert profile document
    profiles_col = get_collection(settings.MONGO_PROFILES_COLLECTION)
    profiles_col.update_one(
        {"user_id": clerk_user_id},
        {
            "$set": {
                "first_name": first_name,
                "last_name": last_name,
                "avatar_url": image_url,
                "updated_at": now,
            },
            "$setOnInsert": {
                "user_id": clerk_user_id,
                "bio": None,
                "phone": None,
                "preferences": {},
                "created_at": now,
            },
        },
        upsert=True,
    )

    # Step 6: Set refresh token as HttpOnly cookie
    _set_refresh_cookie(response, refresh_token)

    # Step 7: Return access token + user info
    print(f"[Auth] Session created for user '{clerk_user_id}' ({email})")

    # Fetch the final user state from DB for response
    user_doc = users_col.find_one(
        {"clerk_user_id": clerk_user_id},
        {"_id": 0, "hashed_refresh_token": 0},
    )

    return SessionResponse(
        access_token=access_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user={
            "id": clerk_user_id,
            "email": email,
            "first_name": first_name,
            "last_name": last_name,
            "avatar_url": image_url,
            "is_verified": user_doc.get("is_verified", False) if user_doc else False,
            "is_banned": user_doc.get("is_banned", False) if user_doc else False,
            "role": user_doc.get("role", "user") if user_doc else "user",
        },
    )


# ── POST /api/auth/refresh ──────────────────────────────────────────────────

@router.post("/refresh")
def refresh_session(request: Request, response: Response):
    """
    Refresh the access token using the refresh token cookie.

    Implements refresh token rotation:
        1. Read refresh token from HttpOnly cookie
        2. Verify it (signature + expiry)
        3. Hash it and compare against the stored hash in MongoDB
        4. Generate a new access token + new refresh token
        5. Update the stored hash, set new cookie
    """
    refresh_token = request.cookies.get("trase_refresh_token")
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No refresh token found",
        )

    # Verify the refresh token
    try:
        payload = verify_refresh_token(refresh_token)
    except jwt.ExpiredSignatureError:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired. Please sign in again.",
        )
    except jwt.InvalidTokenError as err:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {err}",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token payload",
        )

    # Check the hash matches what's stored in MongoDB
    users_col = get_collection(settings.MONGO_USERS_COLLECTION)
    user_doc = users_col.find_one({"clerk_user_id": user_id})

    if not user_doc:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if user_doc.get("is_banned"):
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been suspended",
        )

    stored_hash = user_doc.get("hashed_refresh_token")
    incoming_hash = hash_token(refresh_token)

    if stored_hash != incoming_hash:
        # Possible token reuse attack — invalidate all tokens
        users_col.update_one(
            {"clerk_user_id": user_id},
            {"$set": {"hashed_refresh_token": None}},
        )
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked. Please sign in again.",
        )

    # Rotate: issue new tokens
    new_access_token = create_access_token(
        user_id,
        user_doc.get("email", ""),
        user_doc.get("role", "user"),
    )
    new_refresh_token = create_refresh_token(user_id)
    new_hashed_refresh = hash_token(new_refresh_token)

    # Update stored hash
    users_col.update_one(
        {"clerk_user_id": user_id},
        {
            "$set": {
                "hashed_refresh_token": new_hashed_refresh,
                "updated_at": datetime.now(tz=timezone.utc),
            }
        },
    )

    # Set new refresh cookie
    _set_refresh_cookie(response, new_refresh_token)

    print(f"[Auth] Token refreshed for user '{user_id}'")

    return {
        "access_token": new_access_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


# ── POST /api/auth/logout ───────────────────────────────────────────────────

@router.post("/logout")
def logout(request: Request, response: Response):
    """
    Logs out the user:
        1. Clear the refresh token cookie
        2. Remove the hashed refresh token from MongoDB
    """
    refresh_token = request.cookies.get("trase_refresh_token")

    if refresh_token:
        try:
            payload = verify_refresh_token(refresh_token)
            user_id = payload.get("sub")
            if user_id:
                users_col = get_collection(settings.MONGO_USERS_COLLECTION)
                users_col.update_one(
                    {"clerk_user_id": user_id},
                    {
                        "$set": {
                            "hashed_refresh_token": None,
                            "updated_at": datetime.now(tz=timezone.utc),
                        }
                    },
                )
                print(f"[Auth] User '{user_id}' logged out")
        except jwt.InvalidTokenError:
            pass  # Token already invalid, just clear cookie

    _clear_refresh_cookie(response)
    return {"message": "Logged out successfully"}


# ── GET /api/auth/me ─────────────────────────────────────────────────────────

@router.get("/me")
def get_current_user_profile(user: dict = Depends(get_current_user)):
    """
    Returns the authenticated user's profile from MongoDB.
    Requires a valid access token in the Authorization header.
    """
    user_id = user.get("sub")

    users_col = get_collection(settings.MONGO_USERS_COLLECTION)
    user_doc = users_col.find_one(
        {"clerk_user_id": user_id},
        {"_id": 0, "hashed_refresh_token": 0},
    )

    profiles_col = get_collection(settings.MONGO_PROFILES_COLLECTION)
    profile_doc = profiles_col.find_one(
        {"user_id": user_id},
        {"_id": 0},
    )

    if not user_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return {
        "user": user_doc,
        "profile": profile_doc,
    }
