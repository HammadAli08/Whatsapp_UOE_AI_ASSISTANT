"""
Reranker Module

Uses BAAI/bge-reranker-base via the HuggingFace Inference API
(text-classification pipeline) to rerank retrieved documents.

The BGE reranker takes [query, passage] pairs and returns relevance scores.

Latency optimizations:
  - Persistent httpx.Client with keep-alive (skip TCP + TLS handshake after 1st call)
  - HTTP/2 enabled for multiplexing
  - Tight 10s timeout to fail fast
  - Graceful fallback to original order on any failure

Robustness:
  - Tries batch request first; if shape mismatch falls back to individual requests
  - Handles every known HF response format
  - Auto-disables for cooldown after repeated failures
"""

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from langsmith import traceable

from .config import (
    HF_API_KEY,
    HF_RERANKER_MODEL,
    HF_INFERENCE_URL,
    HF_API_TIMEOUT,
    RERANKER_FAILURE_COOLDOWN_SECONDS,
    DEFAULT_TOP_K_RERANK,
)

logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranks retrieved documents using HuggingFace Inference API.

    The httpx.Client is created once and reused across all requests
    (TCP keep-alive, connection pooling, pre-set auth header).

    Strategy:
      1. Try batch request (all pairs at once)
      2. If batch returns wrong number of scores → fallback to individual requests
      3. On repeated failures → disable for cooldown period
    """

    def __init__(self):
        self.api_key = HF_API_KEY
        self.model = HF_RERANKER_MODEL
        self.url = f"{HF_INFERENCE_URL}/{self.model}"
        # Persistent client — keeps TCP + TLS alive across requests
        self._client = httpx.Client(
            http2=True,
            timeout=httpx.Timeout(HF_API_TIMEOUT, connect=5.0),
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=120),
        )
        self._disabled_until = 0.0
        self._consecutive_failures = 0
        logger.info("Reranker initialized: %s (persistent client)", self.model)

    # ── Score extraction helpers ─────────────────────────────────────

    @staticmethod
    def _extract_relevance_score(result: Any) -> Optional[float]:
        """Extract a single relevance score from various HF response shapes."""
        if isinstance(result, (int, float)):
            return float(result)

        if isinstance(result, dict):
            label = str(result.get("label", ""))
            raw_score = result.get("score")
            if raw_score is None:
                return None
            score = float(raw_score)
            if label == "LABEL_0":
                # In binary setup, LABEL_0 is typically "not relevant".
                return 1.0 - score
            return score

        if isinstance(result, list):
            label_one = None
            best = None
            for item in result:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", ""))
                score = float(item.get("score", 0.0))
                if best is None or score > best:
                    best = score
                if label == "LABEL_1":
                    label_one = score
            if label_one is not None:
                return label_one
            return best

        return None

    def _parse_response_scores(
        self, results: Any, expected: int
    ) -> Optional[List[float]]:
        """Parse reranker API response into a flat list of relevance scores.

        Handles multiple known HF Inference API response shapes:
        1. List of dicts:   [{"label":"LABEL_1","score":0.9}, ...]
        2. Nested list:     [[{"label":"LABEL_0","score":0.1}, {"label":"LABEL_1","score":0.9}], ...]
        3. List of floats:  [0.9, 0.3, ...]
        4. Dict with key:   {"scores": [0.9, 0.3, ...]}
        """
        if not isinstance(results, (list, dict)):
            return None

        # Shape 4: dict wrapper
        if isinstance(results, dict):
            inner = results.get("scores") or results.get("results")
            if isinstance(inner, list):
                results = inner
            else:
                return None

        if not results:
            return None

        scores: List[float] = []

        for item in results:
            # Shape 3: bare float/int
            if isinstance(item, (int, float)):
                scores.append(float(item))
                continue

            # Shape 1: single dict per pair
            if isinstance(item, dict):
                s = self._extract_relevance_score(item)
                if s is not None:
                    scores.append(s)
                    continue
                return None

            # Shape 2: list of label-dicts per pair (binary classifier output)
            if isinstance(item, list):
                s = self._extract_relevance_score(item)
                if s is not None:
                    scores.append(s)
                    continue
                return None

            # Unknown element type → bail out
            return None

        return scores if scores else None

    # ── Individual reranking fallback ────────────────────────────────

    def _rerank_individual(
        self, query: str, texts: List[str]
    ) -> Optional[List[float]]:
        """
        Fallback: send one request per (query, passage) pair.

        Slower but avoids batch response shape issues. Uses the same
        persistent httpx client so keep-alive helps latency.
        """
        scores: List[float] = []
        for i, text in enumerate(texts):
            try:
                pair = {"text": query, "text_pair": text}
                payload = {"inputs": pair}  # Single pair object
                resp = self._client.post(self.url, json=payload)
                if resp.status_code != 200:
                    logger.warning(
                        "Individual rerank %d/%d HTTP %d",
                        i + 1, len(texts), resp.status_code,
                    )
                    return None  # Bail — don't partially score
                result = resp.json()
                score = self._extract_relevance_score(result)
                if score is None:
                    # Try unwrapping one level (some models nest in a list)
                    if isinstance(result, list) and len(result) == 1:
                        score = self._extract_relevance_score(result[0])
                    if score is None:
                        logger.warning(
                            "Individual rerank %d/%d: unparseable result: %s",
                            i + 1, len(texts), str(result)[:200],
                        )
                        return None
                scores.append(score)
            except Exception as exc:
                logger.warning("Individual rerank %d/%d error: %s", i + 1, len(texts), exc)
                return None
        return scores

    # ── Failure tracking ─────────────────────────────────────────────

    def _mark_success(self) -> None:
        self._consecutive_failures = 0
        self._disabled_until = 0.0

    def _mark_failure(self, reason: str) -> None:
        self._consecutive_failures += 1
        logger.warning("Reranker failure (%d): %s", self._consecutive_failures, reason)
        if self._consecutive_failures >= 3:
            self._disabled_until = time.time() + RERANKER_FAILURE_COOLDOWN_SECONDS
            logger.warning(
                "Reranker temporarily disabled for %.0fs",
                RERANKER_FAILURE_COOLDOWN_SECONDS,
            )

    # ── Main rerank method ───────────────────────────────────────────
    @traceable(name="reranker.rerank", run_type="chain")
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = DEFAULT_TOP_K_RERANK,
    ) -> List[Dict]:
        """
        Rerank documents by relevance to query via HF Inference API.

        Strategy:
          1. Try batch request with all pairs
          2. If batch gives wrong score count → fallback to individual requests
          3. If individual also fails → return docs in original order

        Args:
            query: The search query
            documents: List of retrieved documents with 'text' field
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents (top_k most relevant)
        """
        if not documents:
            return []

        if not self.api_key:
            return documents[:top_k]

        if time.time() < self._disabled_until:
            logger.debug("Reranker disabled (cooldown), returning original order")
            return documents[:top_k]

        try:
            # Extract text content from documents
            texts = []
            for doc in documents:
                text = doc.get("text", "")
                if not text:
                    text = doc.get("page_content", "")
                if not text:
                    text = doc.get("metadata", {}).get("text_preview", "")
                texts.append(text if text else "")

            scores = None

            # ── Strategy 1: Batch request ────────────────────────────
            # Use proper HF text-classification pair format
            pairs = [{"text": query, "text_pair": text} for text in texts]
            payload = {"inputs": pairs}

            response = self._client.post(self.url, json=payload)

            if response.status_code == 200:
                results = response.json()
                batch_scores = self._parse_response_scores(results, len(documents))
                if batch_scores is not None and len(batch_scores) == len(documents):
                    scores = batch_scores
                    logger.debug("Batch rerank succeeded: %d scores", len(scores))
                else:
                    logger.info(
                        "Batch rerank shape mismatch (expected %d, got %s) — "
                        "falling back to individual requests",
                        len(documents),
                        len(batch_scores) if batch_scores else "None",
                    )
            else:
                logger.info(
                    "Batch rerank HTTP %d — falling back to individual requests",
                    response.status_code,
                )

            # ── Strategy 2: Individual requests (fallback) ───────────
            if scores is None:
                scores = self._rerank_individual(query, texts)

            # ── Apply scores or give up ──────────────────────────────
            if scores is not None and len(scores) == len(documents):
                scored_docs = list(zip(documents, scores))
                scored_docs.sort(key=lambda x: float(x[1]), reverse=True)
                self._mark_success()
                return [doc for doc, _ in scored_docs[:top_k]]

            # Both strategies failed
            self._mark_failure(
                f"could not obtain {len(documents)} scores "
                f"(batch + individual both failed)"
            )
            return documents[:top_k]

        except Exception as exc:
            self._mark_failure(str(exc))
            return documents[:top_k]


# Singleton instance
_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    """Get or create Reranker singleton."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
