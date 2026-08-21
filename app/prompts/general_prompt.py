"""
general_prompt.py
──────────────────
Defines the ChatPromptTemplate used by GeneralLLM handler.

Runtime variables:
  - {chat_history} : multi-turn conversation history
  - {question}     : the user's current message
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from app.prompts.system_prompt import SYSTEM_PROMPT

GENERAL_HUMAN_TEMPLATE = """\
Recent Chat History:
{chat_history}

User Message:
{question}

Instructions:
- Respond naturally, warmly, and concisely as Trase AI.
- Use conversation history to accurately answer questions about previous turns (e.g. "What bus did I choose?", "Which route were we discussing?").
- If the user asks for bus routes or schedules, politely suggest rephrasing with origin and destination.

Answer:"""

GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(GENERAL_HUMAN_TEMPLATE),
])
