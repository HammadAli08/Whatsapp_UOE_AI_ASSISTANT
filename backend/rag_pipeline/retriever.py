"""
Retriever Module

Handles vector search against Pinecone with namespace isolation.
"""

import copy
import logging
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pinecone import Pinecone

from langsmith import traceable

from .config import (
    CACHE_MAX_ENTRIES,
    EMBEDDING_CACHE_TTL_SECONDS,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_DIMENSIONS,
    RETRIEVAL_CACHE_TTL_SECONDS,
    DEFAULT_TOP_K_RETRIEVE,
)

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant documents from Pinecone vector store.
    Enforces namespace isolation.
    """

    def __init__(self):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

        # Initialize OpenAI client once (keep-alive, pooled connections)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

        # Small in-memory caches to absorb repeated queries in short windows.
        self._embedding_cache: "OrderedDict[str, Tuple[float, List[float]]]" = OrderedDict()
        self._retrieval_cache: "OrderedDict[Tuple[str, str, int], Tuple[float, List[Dict]]]" = OrderedDict()
        self._cache_lock = Lock()

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(query.strip().lower().split())

    def _cache_get(self, cache: OrderedDict, key: Any, ttl_seconds: float):
        now = time.time()
        with self._cache_lock:
            payload = cache.get(key)
            if payload is None:
                return None
            ts, value = payload
            if now - ts > ttl_seconds:
                cache.pop(key, None)
                return None
            cache.move_to_end(key)
            return copy.deepcopy(value)

    def _cache_set(self, cache: OrderedDict, key: Any, value: Any) -> None:
        with self._cache_lock:
            cache[key] = (time.time(), copy.deepcopy(value))
            cache.move_to_end(key)
            while len(cache) > CACHE_MAX_ENTRIES:
                cache.popitem(last=False)

    @traceable(name="retriever.embed_query", run_type="embedding")
    def _embed_query(self, query: str) -> List[float]:
        normalized = self._normalize_query(query)
        cached = self._cache_get(self._embedding_cache, normalized, EMBEDDING_CACHE_TTL_SECONDS)
        if cached is not None:
            return cached

        request_kwargs: Dict[str, Any] = {
            "model": OPENAI_EMBEDDING_MODEL,
            "input": query,
        }
        if OPENAI_EMBEDDING_DIMENSIONS > 0:
            request_kwargs["dimensions"] = OPENAI_EMBEDDING_DIMENSIONS

        response = self.openai_client.embeddings.create(**request_kwargs)
        embedding = response.data[0].embedding
        self._cache_set(self._embedding_cache, normalized, embedding)
        return embedding

    @traceable(name="retriever.retrieve", run_type="retriever")
    def retrieve(
        self,
        query: str,
        namespace: str,
        top_k: int = DEFAULT_TOP_K_RETRIEVE
    ) -> List[Dict]:
        """
        Retrieve relevant documents from Pinecone.
        
        Args:
            query: The search query
            namespace: Pinecone namespace to search in
            top_k: Number of results to retrieve
            
        Returns:
            List of document dictionaries with id, score, text, metadata
        """
        normalized = self._normalize_query(query)
        cache_key = (namespace, normalized, int(top_k))
        cached_docs = self._cache_get(self._retrieval_cache, cache_key, RETRIEVAL_CACHE_TTL_SECONDS)
        if cached_docs is not None:
            return cached_docs

        # Generate embedding for query
        t_embed = time.perf_counter()
        query_embedding = self._embed_query(query)
        embed_seconds = time.perf_counter() - t_embed

        # Search in specific namespace only (strict isolation)
        t_query = time.perf_counter()
        results = self.index.query(
            vector=query_embedding,
            namespace=namespace,
            top_k=top_k,
            include_metadata=True
        )
        query_seconds = time.perf_counter() - t_query

        # Convert to document format
        documents = []
        for match in results.matches:
            metadata = match.metadata or {}

            # Extract text content (try multiple fields)
            text = metadata.get("text_preview", "")
            if not text:
                text = metadata.get("page_content", "")

            doc = {
                "id": match.id,
                "score": float(match.score),
                "text": text,
                "metadata": metadata
            }
            documents.append(doc)

        self._cache_set(self._retrieval_cache, cache_key, documents)
        logger.debug(
            "Retriever timings: embed=%.2fs pinecone=%.2fs top_k=%d namespace=%s",
            embed_seconds,
            query_seconds,
            top_k,
            namespace,
        )
        return documents


# Singleton instance
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Get or create Retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
