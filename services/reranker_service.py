"""Infomaniak Cohere-compatible remote reranker."""

import logging
from typing import List

import httpx
from langchain_core.documents.base import Document

logger = logging.getLogger(__name__)


class InfomaniakReranker:
    """Calls Infomaniak's /cohere/v2/rerank endpoint to score and filter documents.

    Scores returned are calibrated probabilities in [0, 1].  Documents below
    `threshold` are dropped; survivors are returned sorted by score descending.
    """

    _ENDPOINT = "https://api.infomaniak.com/2/ai/{product_id}/cohere/v2/rerank"

    def __init__(self, api_key: str, product_id: str, model: str, threshold: float):
        self._api_key = api_key
        self._url = self._ENDPOINT.format(product_id=product_id)
        self._model = model
        self._threshold = threshold

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Return documents filtered by threshold and sorted by relevance score (desc)."""
        if not documents:
            return []

        payload = {
            "model": self._model,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "return_documents": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(self._url, json=payload, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(
                f"Infomaniak reranker API error {response.status_code}: {response.text}"
            )

        results = response.json().get("results", [])
        scored = [(r["relevance_score"], documents[r["index"]]) for r in results]
        passing = [(score, doc) for score, doc in scored if score >= self._threshold]
        passing.sort(key=lambda x: x[0], reverse=True)

        if passing:
            logger.info(
                f"remote rerank: {len(documents)} candidates → {len(passing)} passed "
                f"threshold={self._threshold} (top score={passing[0][0]:.3f})"
            )
        else:
            logger.info(
                f"remote rerank: {len(documents)} candidates → 0 passed "
                f"threshold={self._threshold}"
            )

        return [doc for _, doc in passing]
