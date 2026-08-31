"""
input_cleaner.py
─────────────────
Sanitization and normalization pipeline for incoming user prompts.

Performs:
1. Whitespace stripping and normalization (removes multiple spaces, tabs, newlines).
2. Control and non-printable character removal.
3. Common query intent normalization (standardizing known city names and travel keywords).
4. Safety check to ensure prompt is non-empty after cleaning.
"""

import re
import unicodedata


def clean_user_prompt(prompt: str) -> str:
    """
    Cleans, sanitizes, and normalizes a user query before it enters
    intent classification, vector search, or LLM generation.

    Args:
        prompt: Raw input string from the client.

    Returns:
        Cleaned and normalized query string.
    """
    if not prompt:
        return ""

    # 1. Normalize unicode characters (NFKC)
    cleaned = unicodedata.normalize("NFKC", prompt)

    # 2. Strip non-printable/control characters (keep basic punctuation and alphanumeric)
    cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch)[0] != "C" or ch in "\n\t")

    # 3. Collapse multiple whitespaces into a single space
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)
    cleaned = cleaned.strip()

    # 4. Standardize common Tamil Nadu city spellings for optimal RAG match
    city_aliases = {
        r"\bchenai\b": "Chennai",
        r"\bmadhurai\b": "Madurai",
        r"\bcoimbator\b": "Coimbatore",
        r"\bkovai\b": "Coimbatore",
        r"\btrichy\b": "Trichy",
        r"\btiruchirappalli\b": "Trichy",
        r"\btirupur\b": "Tiruppur",
        r"\btindivanam\b": "Tindivanam",
        r"\bvillupuram\b": "Villupuram",
    }

    for pattern, replacement in city_aliases.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    return cleaned
