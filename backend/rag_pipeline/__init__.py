"""
RAG Pipeline Package

Production-grade RAG pipeline for UOE Lahore Academic AI Assistant.

Components:
- QueryEnhancer: Optimizes queries for vector search
- Retriever: Fetches documents from Pinecone
- Generator: Produces final answers using GPT-4o
- ConversationMemory: Redis-backed short-term session memory
- SmartRAGProcessor: Self-correcting retrieval with best-effort answering (3 retries)
- RAGPipeline: Orchestrates the full pipeline
"""

from .config import VALID_NAMESPACES, NAMESPACE_MAP
from .query_enhancer import QueryEnhancer, get_query_enhancer
from .retriever import Retriever, get_retriever
from .generator import Generator, get_generator
from .memory import ConversationMemory, get_memory
from .smart_rag import (
    SmartRAGProcessor,
    SmartChunkGrader,
    SmartQueryRewriter,
    get_smart_processor,
)
from .pipeline import RAGPipeline, get_pipeline

__all__ = [
    # Configuration
    "VALID_NAMESPACES",
    "NAMESPACE_MAP",

    # Classes
    "QueryEnhancer",
    "Retriever",
    "Generator",
    "ConversationMemory",
    "SmartRAGProcessor",
    "SmartChunkGrader",
    "SmartQueryRewriter",
    "RAGPipeline",

    # Factory functions
    "get_query_enhancer",
    "get_retriever",
    "get_generator",
    "get_memory",
    "get_smart_processor",
    "get_pipeline",
]
