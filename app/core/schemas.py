"""
schemas.py
──────────
Pydantic data models defining the exact data storage schemas across MongoDB
collections in Trase Bus Travel API.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


# ── 1. File Registry Schema ──────────────────────────────────────────────────

class FileRegistryEntry(BaseModel):
    """Schema for document in file_registry collection."""
    filename: str = Field(..., description="Unique source file name")
    file_hash: str = Field(..., description="MD5 hex checksum of source file")
    chunk_count: int = Field(..., ge=0, description="Total chunks ingested into vector store")
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp of ingestion"
    )


# ── 2. Vector Store Metadata Schema ──────────────────────────────────────────

class ChunkLineLoc(BaseModel):
    lines_from: Optional[int] = Field(None, alias="from")
    lines_to: Optional[int] = Field(None, alias="to")


class VectorChunkMetadata(BaseModel):
    """Schema for metadata embedded in bus_routes_v2 documents."""
    source: str = Field(..., description="Source file name")
    file_hash: str = Field(..., description="MD5 digest of source file")
    chunk_id: Optional[int] = Field(None, description="Sequential index of chunk")
    loc: Optional[Dict[str, Any]] = None


class VectorDocument(BaseModel):
    """Representation of a document chunk in bus_routes_v2 vector store."""
    page_content: str = Field(..., description="Raw text snippet of document chunk")
    embedding: List[float] = Field(..., description="Vector embedding representation")
    metadata: VectorChunkMetadata


# ── 3. Chat Session & Message Schemas ────────────────────────────────────────

class QueryMode(str, Enum):
    GENERAL_LLM = "general_llm"
    RAG = "rag"
    TOOL = "tool"
    RAG_AND_TOOL = "rag_and_tool"


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Schema for an individual message within a conversation."""
    message_id: str = Field(..., description="UUID v4 message identifier")
    role: Role = Field(..., description="Role of message sender")
    content: str = Field(..., description="Text content of message")
    mode: Optional[QueryMode] = Field(None, description="Router execution mode")
    tool_called: Optional[str] = Field(None, description="Name of tool invoked if any")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp"
    )


class ChatSession(BaseModel):
    """Schema for session document in chat_sessions collection."""
    session_id: str = Field(..., description="UUID v4 session identifier")
    user_id: Optional[str] = Field(None, description="Optional authenticated user ID")
    messages: List[ChatMessage] = Field(default_factory=list, description="Array of messages in session")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="Session creation UTC timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="Session last updated UTC timestamp"
    )


# ── 4. Structured Bus Route Schemas ──────────────────────────────────────────

class BusStop(BaseModel):
    """Schema for individual bus stop on a route."""
    stop_name: str = Field(..., description="Name of bus stop/terminal")
    sequence_order: int = Field(..., ge=1, description="Stop sequence on route")
    fare_from_origin: float = Field(..., ge=0.0, description="Cumulative fare from origin stop")


class OperatingHours(BaseModel):
    """Operating hours and service frequency."""
    first_bus: str = Field(..., description="First departure time (HH:MM format)")
    last_bus: str = Field(..., description="Last departure time (HH:MM format)")
    frequency_mins: int = Field(..., gt=0, description="Bus departure interval in minutes")


class FareStructure(BaseModel):
    """Pricing calculation parameters."""
    base_fare: float = Field(..., ge=0.0, description="Base ticket fare")
    per_km_rate: float = Field(..., ge=0.0, description="Per-kilometer fare rate")


class BusRoute(BaseModel):
    """Schema for document in structured_bus_routes collection."""
    route_id: str = Field(..., description="Unique route identifier (e.g., R-101)")
    route_number: str = Field(..., description="Public route identifier (e.g., 21B)")
    origin: str = Field(..., description="Originating bus terminal")
    destination: str = Field(..., description="Terminating bus terminal")
    stops: List[BusStop] = Field(default_factory=list, description="Ordered list of stops")
    operating_hours: OperatingHours
    fare_structure: FareStructure


# ── 5. User Schema ──────────────────────────────────────────────────────────

class UserDocument(BaseModel):
    """Schema for document in users collection — stores auth & account state."""
    clerk_user_id: str = Field(..., description="Primary key from Clerk (e.g., user_2x...)")
    email: str = Field(..., description="User email address")
    hashed_refresh_token: Optional[str] = Field(None, description="SHA-256 hash of current refresh token")
    is_verified: bool = Field(False, description="Whether the user's email is verified")
    is_banned: bool = Field(False, description="Whether the user account is banned")
    role: str = Field("user", description="User role: 'user' or 'admin'")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="Account creation UTC timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="Last update UTC timestamp"
    )


# ── 6. Profile Schema ───────────────────────────────────────────────────────

class ProfileDocument(BaseModel):
    """Schema for document in profiles collection — stores user profile info."""
    user_id: str = Field(..., description="References clerk_user_id in users collection")
    first_name: Optional[str] = Field(None, description="User's first name")
    last_name: Optional[str] = Field(None, description="User's last name")
    avatar_url: Optional[str] = Field(None, description="URL to profile avatar image")
    bio: Optional[str] = Field(None, description="Short biography / about me text")
    phone: Optional[str] = Field(None, description="Phone number")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences (theme, language, etc.)")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="Profile creation UTC timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="Last update UTC timestamp"
    )


# ── 7. Connector Schema ─────────────────────────────────────────────────────

class ConnectorDocument(BaseModel):
    """Schema for document in connectors collection — stores external service links."""
    user_id: str = Field(..., description="References clerk_user_id in users collection")
    provider: str = Field(..., description="Service provider name (e.g., 'github', 'google')")
    provider_user_id: Optional[str] = Field(None, description="User's ID on the external provider")
    display_name: Optional[str] = Field(None, description="Display name for this connection")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific metadata")
    is_active: bool = Field(True, description="Whether this connection is currently active")
    connected_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="Connection creation UTC timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="Last update UTC timestamp"
    )

