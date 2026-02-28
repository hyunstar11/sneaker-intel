"""RAG analyst: retrieves relevant data context and generates grounded answers
using the Anthropic Claude API."""

from __future__ import annotations

import os
from dataclasses import dataclass

from sneaker_intel.rag.documents import Document
from sneaker_intel.rag.retriever import Retriever

SYSTEM_PROMPT = """You are a senior demand intelligence analyst specialising in footwear and sneaker markets.
You have access to a knowledge base built from:
- StockX 2019 resale transaction data (99K+ transactions, Yeezy + Off-White)
- Market 2023 dataset (1,994 products across 23 brands)
- Reddit sneaker community data (5,796 posts + comments, 9 subreddits, Feb 2026)

Answer questions using ONLY the provided data context. Be specific — cite numbers.
If the context does not contain enough information to answer confidently, say so clearly
rather than speculating. Keep answers concise (3–5 sentences) and actionable."""


@dataclass
class AnalystResponse:
    question: str
    answer: str
    sources: list[Document]
    context_used: str
    has_llm: bool


class Analyst:
    """End-to-end RAG pipeline: retrieve → augment → generate."""

    MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, retriever: Retriever, api_key: str | None = None):
        self.retriever = retriever
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
        if self._api_key:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self._api_key)

    @property
    def has_llm(self) -> bool:
        return self._client is not None

    def ask(self, question: str, k: int = 4) -> AnalystResponse:
        """Retrieve relevant context and generate a grounded answer."""
        results = self.retriever.retrieve(question, k=k)
        context = self.retriever.format_context(results)
        sources = [doc for doc, _ in results]

        if not self.has_llm:
            answer = (
                "⚠️  ANTHROPIC_API_KEY not set — LLM generation disabled.\n"
                "Retrieved context is shown below. Set the env var to enable answers."
            )
        else:
            user_message = (
                f"Data context:\n\n{context}\n\n"
                f"Question: {question}"
            )
            response = self._client.messages.create(
                model=self.MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            answer = response.content[0].text

        return AnalystResponse(
            question=question,
            answer=answer,
            sources=sources,
            context_used=context,
            has_llm=self.has_llm,
        )
