"""
LangChain wrapper around Google Gemini 1.5 Flash
"""

import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from .models import RetrievedChunk, Answer

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.0,
)

SYSTEM_PROMPT = (
    "You are a concise HSC Bangla assistant. "
    "Carefully read the user question and the provided context. "
    "Answer **exactly** what the question asks (name, number, phrase, sentence, or short paragraph). "
    "Do not add extra explanations unless the question itself requests them. "
    "Always reply in the same language as the question. "
    "If the answer is not present, say 'আমি নিশ্চিত নই'."
)


def generate_answer(query: str, chunks: List[RetrievedChunk]) -> Answer:
    """
    Generate a concise answer from retrieved chunks.
    """
    context = "\n\n".join(c.text for c in chunks)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]

    response = llm.invoke(messages)
    return Answer(
        text=response.content.strip(),
        sources=[c.metadata for c in chunks],
    )