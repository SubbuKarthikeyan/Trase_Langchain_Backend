"""
rag_prompt.py
──────────────
Defines the RAG ChatPromptTemplate used inside the LCEL streaming chain.

The template receives two variables at runtime:
  - {context}  : retrieved document chunks joined into a single string
  - {question} : the user's original question
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from app.prompts.system_prompt import SYSTEM_PROMPT

# ──────────────────────────────────────────────────────────────────────────────
# RAG prompt template (LangChain LCEL compatible)
# ──────────────────────────────────────────────────────────────────────────────

RAG_HUMAN_TEMPLATE = """\
Context from Travel Database:
{context}

User Question:
{question}

Instructions:
- Answer the user's question using ONLY the provided database context.
- MULTIPLE OPTIONS: If asked for available buses or if multiple buses match, list them clearly as separate options (Option 1, Option 2...) with schedules, bus type, facilities, and fares.
- STOP FARE CALCULATION: When calculating stop fares, 1st stop costs Minimum Fare (`MinFare`). Subsequent additional stops cost `MinFare + (additional_stops * AddStopFare)`. Show the step-by-step fare breakdown.
- PRIVACY & DATA SANITIZATION: Never mention internal registration IDs (e.g. TN31 AB 1001), route codes (e.g. R001), database _ids, or chunk hashes.
- If details are not available in the context, politely reply:
  "I'm sorry, but those details are not available in our travel database at the moment. Please let me know if there's anything else I can help you find!"

Answer:"""


RAG_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(RAG_HUMAN_TEMPLATE),
])
