"""
Smart RAG Package — Self-Correcting Retrieval with Best-Effort Answering

Provides:
  - SmartRAGProcessor: Orchestrates the grade → rewrite → retry loop (6 attempts)
  - SmartChunkGrader: Grades chunks with confidence scoring (0.0–1.0)
  - SmartQueryRewriter: Rewrites failed queries with 6-level strategy escalation
  - get_smart_processor(): Singleton factory
  - get_fallback_message(): Graceful fallback text (only when zero chunks exist)
  - SMART_RAG_CONFIG: Configuration constants
"""

from .config import SMART_RAG_CONFIG
from .grader import SmartChunkGrader
from .rewriter import SmartQueryRewriter
from .processor import (
    SmartRAGProcessor,
    get_smart_processor,
    get_fallback_message,
)

__all__ = [
    "SMART_RAG_CONFIG",
    "SmartChunkGrader",
    "SmartQueryRewriter",
    "SmartRAGProcessor",
    "get_smart_processor",
    "get_fallback_message",
]
