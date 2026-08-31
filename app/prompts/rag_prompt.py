"""
rag_prompt.py
──────────────
Defines the RAG ChatPromptTemplate with Chat History, point-based formatting,
route progression display, and stop-based fare calculation.

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
- Use the retrieved database context to answer accurately.
- When listing bus choices, present them clearly as numbered point-based options (Option 1, Option 2, etc.) including Bus Name, Bus Type, Schedule, Seats, Facilities, Full Route Progression, and Base MinFare / Additional Fare.
- For fare calculations, use the formula: Total Fare = MinFare + (Number of Intermediate Stops × Additional Fare per Stop) and show the calculation breakdown clearly.
- When the user selects or finalizes a bus, provide the finalized journey summary and ask if they'd like it emailed.
- If details are not available in context, politely state:
  "I'm sorry, but those details are not available in our travel database right now."

Answer:"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(RAG_HUMAN_TEMPLATE),
])