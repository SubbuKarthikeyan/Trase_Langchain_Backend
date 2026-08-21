"""
session_memory.py
──────────────────
In-memory conversation session cache for Trase Bus Travel API.

Maintains multi-turn conversation history, user selections, and pending email
tool consent flags. Serves as a lightweight in-memory cache preparing for future
LangGraph state checkpointer integration.
"""

from typing import List, Dict, Optional


class SessionMemory:
    """
    In-memory session state holder for active chat turns.
    """
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.last_selected_bus: Optional[str] = None
        self.pending_email_consent: bool = False
        self.user_email: Optional[str] = None

    def add_user_message(self, content: str) -> None:
        """Appends a user message to history."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Appends an assistant message to history."""
        self.messages.append({"role": "assistant", "content": content})

    def get_history_text(self, max_messages: int = 10) -> str:
        """
        Formats recent messages into a clean conversation string
        suitable for LLM prompt context injection.
        """
        recent = self.messages[-max_messages:]
        if not recent:
            return "No previous conversation context."

        formatted_lines = []
        for msg in recent:
            role_label = "User" if msg["role"] == "user" else "Trase AI"
            formatted_lines.append(f"{role_label}: {msg['content']}")
        
        return "\n".join(formatted_lines)

    def set_selected_bus(self, bus_info: str) -> None:
        """Records the bus option selected by the user."""
        self.last_selected_bus = bus_info

    def get_selected_bus(self) -> Optional[str]:
        """Retrieves the last selected bus info."""
        return self.last_selected_bus

    def clear(self) -> None:
        """Resets session memory."""
        self.messages.clear()
        self.last_selected_bus = None
        self.pending_email_consent = False
        self.user_email = None


# Global singleton in-memory session store (can be indexed by session_id in future)
session_store = SessionMemory()
