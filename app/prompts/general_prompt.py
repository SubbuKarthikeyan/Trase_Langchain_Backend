"""
general_prompt.py
──────────────────
Defines the ChatPromptTemplate used by the GeneralLLM handler.

Used when the router determines a query does NOT need RAG context
(e.g., greetings, general travel advice, small talk).

The template receives one variable at runtime:
  - {question}: the user's message
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from app.prompts.system_prompt import SYSTEM_PROMPT

# ──────────────────────────────────────────────────────────────────────────────
# General LLM prompt (no RAG context injected)
# ──────────────────────────────────────────────────────────────────────────────

GENERAL_HUMAN_TEMPLATE = """\
User Message:
{question}

Instructions:
- Respond naturally, warmly, and helpfully as Trase, the travel assistant.
- If the user is asking a general travel-related question not specific to bus \
routes or schedules, you may answer from general knowledge.
- If the user is asking for specific bus schedules, routes, or fares, \
politely let them know you need to check the database and suggest they rephrase \
their question as a route or destination query.
- Never fabricate specific bus times, fares, or route details.

Answer:"""


GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(GENERAL_HUMAN_TEMPLATE),
])
