"""TF-IDF retriever with cosine similarity.

In production this would use dense embeddings (e.g. text-embedding-3-small)
stored in a vector DB (Chroma, Pinecone). TF-IDF is used here to keep the
system dependency-free while demonstrating the same retrieval interface.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sneaker_intel.rag.documents import Document


class Retriever:
    """Index a collection of documents and retrieve the top-k most relevant
    given a natural-language query."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_df=0.95,
            min_df=1,
        )
        corpus = [f"{doc.title}. {doc.content}" for doc in documents]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Return the top-k documents with their cosine similarity scores."""
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.documents[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def format_context(self, results: list[tuple[Document, float]]) -> str:
        """Format retrieved documents as a context string for the LLM prompt."""
        parts = []
        for i, (doc, score) in enumerate(results, 1):
            parts.append(
                f"[Source {i} | relevance={score:.3f}]\n"
                f"Title: {doc.title}\n"
                f"{doc.content}"
            )
        return "\n\n---\n\n".join(parts)
