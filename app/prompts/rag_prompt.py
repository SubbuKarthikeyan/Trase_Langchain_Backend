"""
rag_prompt.py
──────────────
Defines the RAG ChatPromptTemplate with Chat History support.

Runtime variables:
  - {chat_history} : multi-turn conversation history
  - {context}      : retrieved document chunks
  - {question}     : the user's current query
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from app.prompts.system_prompt import SYSTEM_PROMPT

RAG_HUMAN_TEMPLATE = """\
Recent Chat History:
{chat_history}

Context from Travel Database:
{context}

User Question:
{question}

Instructions:
- Answer concisely using the provided context and conversation history.
- When listing bus choices, present them clearly as numbered options (Option 1, Option 2...) with departure, arrival, bus type, fares, and facilities.
- When a user chooses a bus, show its details and ask: "Would you like me to send these bus details to your email?"
- If details are not available in context, politely reply:
  "I'm sorry, but those details are not available in our travel database right now."

Answer:"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(RAG_HUMAN_TEMPLATE),
])