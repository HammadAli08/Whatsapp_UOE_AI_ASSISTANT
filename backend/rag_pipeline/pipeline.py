"""
RAG Pipeline Orchestrator

Main pipeline class that orchestrates all RAG components with short-term
conversation memory, streaming, LangSmith tracing, and optional Smart RAG.

Smart RAG (when enabled via ``enable_smart=True``):
  1. Enhance the query, retrieve chunks from vector DB
  2. LLM grades every chunk for relevance
  3. If chunks are good enough → generate answer
  4. If not → rewrite query and re-retrieve (up to 6 retries)
  5. Accumulates ALL relevant chunks found across every iteration
  6. Early exit when enough high-quality chunks collected
  7. After 6 retries → answer with ALL relevant chunks collected (best-effort)
  8. If very few chunks → detect if clarification from user would help
  9. Only uses "sorry" fallback when literally zero chunks exist

When Smart RAG is disabled the pipeline works as standard single-step RAG:
  retrieve → generate.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Generator as Gen

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from .config import (
    NAMESPACE_MAP,
    DEFAULT_TOP_K_RETRIEVE,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
)
from .query_enhancer import get_query_enhancer
from .retriever import get_retriever
from .generator import get_generator
from .memory import get_memory
from .smart_rag import (
    get_smart_processor,
    get_fallback_message as get_smart_fallback_message,
    SMART_RAG_CONFIG,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# HELPER: build sources list from documents
# ═════════════════════════════════════════════════════════════════════════════

def _extract_sources(docs: List[Dict]) -> List[Dict]:
    sources = []
    for doc in docs:
        m = doc.get("metadata", {})
        sources.append({
            "file": m.get("source_file", "Unknown"),
            "page": m.get("page_number", "N/A"),
            "score": doc.get("score", 0),
            "course_code": m.get("course_code", ""),
            "department": m.get("department", ""),
        })
    return sources


# ═════════════════════════════════════════════════════════════════════════════
# RAG PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """Production RAG pipeline with namespace isolation, memory, streaming,
    and optional Smart RAG (self-correcting retrieval with best-effort answering)."""

    def __init__(self):
        self.query_enhancer = get_query_enhancer()
        self.retriever = get_retriever()
        self.generator = get_generator()
        self.memory = get_memory()
        self.smart_processor = get_smart_processor()

    # ── Namespace resolution ─────────────────────────────────────────

    def _resolve_namespace(self, namespace: str) -> str:
        if namespace in NAMESPACE_MAP:
            return NAMESPACE_MAP[namespace]
        if namespace in NAMESPACE_MAP.values():
            return namespace
        valid = list(NAMESPACE_MAP.keys())
        raise ValueError(f"Invalid namespace '{namespace}'. Valid options: {valid}")

    # ── Smart retrieval loop (best-effort) ────────────────────

    @traceable(name="rag_pipeline.smart_retrieve", run_type="chain")
    def _smart_retrieve(
        self,
        user_query: str,
        enhanced_query: str,
        pinecone_namespace: str,
        top_k: int,
    ) -> Dict:
        """
        Self-correcting retrieval with best-effort answering:
          attempt 0  → retrieve with enhanced_query, grade chunks
          attempt 1+ → rewrite query, retrieve again, grade again
          After 6 retries → answer with ALL relevant chunks collected
          If few chunks → check if user should provide more details
          Zero chunks ever → return fallback "sorry" message

        Returns dict with keys: documents, metrics, query_used, clarification
        """
        proc = self.smart_processor
        max_retries = SMART_RAG_CONFIG["max_retries"]
        boost = SMART_RAG_CONFIG["retry_top_k_boost"]

        current_query = enhanced_query
        all_relevant: List[Dict] = []  # Accumulate across ALL attempts
        all_irrelevant: List[Dict] = []
        total_retrievals = 0
        total_graded = 0
        rewrites: List[Dict] = []
        seen_ids = set()  # Deduplicate chunks across attempts

        for attempt in range(max_retries + 1):
            # Retrieve — use more chunks on retries (progressively)
            retrieve_k = top_k + (boost * attempt)
            documents = self.retriever.retrieve(
                query=current_query, namespace=pinecone_namespace, top_k=retrieve_k,
            )
            total_retrievals += 1

            if not documents:
                logger.info("Smart attempt %d: zero documents retrieved", attempt)
                if attempt < max_retries:
                    current_query = proc.rewrite_query(user_query, all_irrelevant, attempt + 1)
                    rewrites.append({"attempt": attempt + 1, "rewritten_query": current_query})
                    continue
                break

            # Grade against user's original question (not enhanced/rewritten)
            # so grading matches user intent, not search-expanded terms
            relevant, irrelevant = proc.grade_chunks(user_query, documents)
            total_graded += len(documents)

            # Accumulate relevant chunks (deduplicate by ID)
            for chunk in relevant:
                chunk_id = chunk.get("id", id(chunk))
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    all_relevant.append(chunk)

            all_irrelevant.extend(irrelevant)

            logger.info(
                "Smart attempt %d: retrieved=%d, relevant=%d, irrelevant=%d, "
                "total_relevant=%d, query='%s'",
                attempt, len(documents), len(relevant), len(irrelevant),
                len(all_relevant), current_query[:80],
            )

            # Check if we accumulated enough good chunks across ALL attempts
            if proc.should_stop_early(all_relevant, attempt):
                logger.info(
                    "Smart RAG early exit: %d relevant chunks accumulated "
                    "(threshold=%d)",
                    len(all_relevant), proc.early_success,
                )
                break

            # Enough relevant chunks in this attempt? → done
            if not proc.should_retry(relevant, attempt):
                break

            # Rewrite for the next attempt
            current_query = proc.rewrite_query(user_query, irrelevant, attempt + 1)
            rewrites.append({"attempt": attempt + 1, "rewritten_query": current_query})

        # ── Post-loop: clarification detection ───────────────────────
        clarification = None
        if len(all_relevant) == 0:
            # Zero chunks: check if we should ask for clarification
            clarification = proc.detect_clarification_needed(
                user_query, all_relevant, all_irrelevant,
            )
        elif len(all_relevant) < proc.min_relevant:
            # Very few chunks: might benefit from user details
            clarification = proc.detect_clarification_needed(
                user_query, all_relevant, all_irrelevant,
            )

        # Best-effort: use ALL relevant chunks collected across attempts
        # Sort by grade_confidence descending so best chunks are fed to generator first
        all_relevant.sort(
            key=lambda c: c.get("grade_confidence", 0.0), reverse=True,
        )

        is_best_effort = (
            len(all_relevant) > 0
            and len(all_relevant) < proc.min_relevant
        )

        metrics = proc.build_metrics(
            total_retrievals=total_retrievals,
            total_chunks_graded=total_graded,
            query_rewrites=rewrites,
            final_relevant_count=len(all_relevant),
            used_fallback=len(all_relevant) == 0,
            best_effort=is_best_effort,
            clarification_asked=clarification is not None,
        )

        logger.info("Smart RAG metrics: %s", json.dumps(metrics))

        return {
            "documents": all_relevant,
            "metrics": metrics,
            "query_used": current_query,
            "clarification": clarification,
        }

    # ── NON-STREAMING QUERY ──────────────────────────────────────────

    @traceable(name="rag_pipeline.query", run_type="chain")
    def query(
        self, user_query: str, namespace: str, enhance_query: bool = True,
        top_k_retrieve: int = DEFAULT_TOP_K_RETRIEVE,
        session_id: str = "",
        enable_smart: bool = False,
    ) -> Dict:
        """Execute the full RAG pipeline (non-streaming)."""
        t_start = time.perf_counter()
        pinecone_namespace = self._resolve_namespace(namespace)

        chat_history = []
        if session_id:
            chat_history = self.memory.get_history(session_id)

        # ── Query enhancement ────────────────────────────────────
        enhanced_query = user_query
        smart_info = None

        if enhance_query:
            t_enhance = time.perf_counter()
            enhanced_query = self.query_enhancer.enhance(user_query, chat_history=chat_history)
            logger.info("⏱ enhance: %.2fs", time.perf_counter() - t_enhance)

        # The retrieval query: use enhanced if available, else raw
        retrieval_query = enhanced_query

        if enable_smart:
            # SMART PATH: self-correcting retrieval loop
            smart_result = self._smart_retrieve(
                user_query, retrieval_query, pinecone_namespace, top_k_retrieve,
            )
            documents = smart_result["documents"]
            smart_info = smart_result["metrics"]
            clarification = smart_result.get("clarification")

            # Fallback only when zero chunks across all attempts
            if not documents:
                # Use clarification message if available, else generic fallback
                fallback = clarification or get_smart_fallback_message()
                if session_id:
                    self.memory.add_turn(session_id, user_query, fallback)
                # Capture run_id even on fallback
                _run_id = None
                try:
                    _rt = get_current_run_tree()
                    if _rt:
                        _run_id = str(_rt.id)
                except Exception:
                    pass
                return {
                    "answer": fallback, "sources": [],
                    "enhanced_query": enhanced_query,
                    "namespace": namespace, "session_id": session_id,
                    "smart_info": smart_info,
                    "run_id": _run_id,
                }
        else:
            # STANDARD PATH: single retrieval
            t_retrieve = time.perf_counter()
            documents = self.retriever.retrieve(
                query=retrieval_query, namespace=pinecone_namespace, top_k=top_k_retrieve,
            )
            logger.info("⏱ retrieve: %.2fs  (docs=%d)", time.perf_counter() - t_retrieve, len(documents))

        if not documents:
            no_result = (
                "No relevant documents found for your query in this namespace. "
                "Please try rephrasing or check if you selected the correct category."
            )
            if session_id:
                self.memory.add_turn(session_id, user_query, no_result)
            _run_id = None
            try:
                _rt = get_current_run_tree()
                if _rt:
                    _run_id = str(_rt.id)
            except Exception:
                pass
            return {
                "answer": no_result, "sources": [],
                "enhanced_query": enhanced_query,
                "namespace": namespace, "session_id": session_id,
                "smart_info": smart_info,
                "run_id": _run_id,
            }

        # ── Use retrieved documents directly (top_k already applied by retriever) ──
        final_docs = documents[:top_k_retrieve]

        # ── Generate ────────────────────────────────────────────────
        t_generate = time.perf_counter()
        answer = self.generator.generate(
            query=user_query, documents=final_docs, namespace=pinecone_namespace,
            chat_history=chat_history, session_id=session_id, enhanced_query=enhanced_query,
        )
        logger.info("⏱ generate: %.2fs", time.perf_counter() - t_generate)

        if session_id:
            self.memory.add_turn(session_id, user_query, answer)

        sources = _extract_sources(final_docs)

        # ── Capture LangSmith run_id for feedback linkage ────────
        run_id = None
        try:
            rt = get_current_run_tree()
            if rt:
                run_id = str(rt.id)
        except Exception:
            pass

        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info("⏱ TOTAL: %.2fs", latency_ms / 1000)

        return {
            "answer": answer, "sources": sources,
            "enhanced_query": enhanced_query,
            "namespace": namespace, "session_id": session_id,
            "smart_info": smart_info,
            "run_id": run_id,
        }

    # ── STREAMING QUERY ──────────────────────────────────────────────

    @traceable(name="rag_pipeline.stream_query", run_type="chain")
    def stream_query(
        self, user_query: str, namespace: str, enhance_query: bool = True,
        top_k_retrieve: int = DEFAULT_TOP_K_RETRIEVE,
        session_id: str = "",
        enable_smart: bool = False,
    ) -> Gen[Dict, None, None]:
        """Execute RAG pipeline with streaming token output."""
        t_start = time.perf_counter()
        pinecone_namespace = self._resolve_namespace(namespace)

        chat_history = []
        if session_id:
            chat_history = self.memory.get_history(session_id)

        # ── Query enhancement ────────────────────────────────────
        enhanced_query = user_query
        smart_info = None

        if enhance_query:
            t_enhance = time.perf_counter()
            enhanced_query = self.query_enhancer.enhance(user_query, chat_history=chat_history)
            logger.info("⏱ enhance: %.2fs", time.perf_counter() - t_enhance)

        # The retrieval query: use enhanced if available, else raw
        retrieval_query = enhanced_query

        if enable_smart:
            # SMART PATH: self-correcting retrieval loop
            smart_result = self._smart_retrieve(
                user_query, retrieval_query, pinecone_namespace, top_k_retrieve,
            )
            documents = smart_result["documents"]
            smart_info = smart_result["metrics"]
            clarification = smart_result.get("clarification")

            # Fallback only when zero chunks
            if not documents:
                fallback = clarification or get_smart_fallback_message()
                if session_id:
                    self.memory.add_turn(session_id, user_query, fallback)
                _run_id = None
                try:
                    _rt = get_current_run_tree()
                    if _rt:
                        _run_id = str(_rt.id)
                except Exception:
                    pass
                yield {
                    "type": "metadata", "sources": [],
                    "enhanced_query": enhanced_query,
                    "namespace": namespace, "session_id": session_id,
                    "smart_info": smart_info,
                    "run_id": _run_id,
                }
                yield {"type": "token", "content": fallback}
                return
        else:
            # STANDARD PATH: single retrieval
            t_retrieve = time.perf_counter()
            documents = self.retriever.retrieve(
                query=retrieval_query, namespace=pinecone_namespace, top_k=top_k_retrieve,
            )
            logger.info("⏱ retrieve: %.2fs  (docs=%d)", time.perf_counter() - t_retrieve, len(documents))

        if not documents:
            no_result = (
                "No relevant documents found for your query in this namespace. "
                "Please try rephrasing or check if you selected the correct category."
            )
            if session_id:
                self.memory.add_turn(session_id, user_query, no_result)
            _run_id = None
            try:
                _rt = get_current_run_tree()
                if _rt:
                    _run_id = str(_rt.id)
            except Exception:
                pass
            yield {
                "type": "metadata", "sources": [],
                "enhanced_query": enhanced_query,
                "namespace": namespace, "session_id": session_id,
                "smart_info": smart_info,
                "run_id": _run_id,
            }
            yield {"type": "token", "content": no_result}
            return

        # ── Use retrieved documents directly (top_k already applied by retriever) ──
        final_docs = documents[:top_k_retrieve]

        sources = _extract_sources(final_docs)

        # ── Capture LangSmith run_id for feedback linkage ────────
        run_id = None
        try:
            rt = get_current_run_tree()
            if rt:
                run_id = str(rt.id)
        except Exception:
            pass

        # ── Emit metadata first ─────────────────────────────────────
        yield {
            "type": "metadata", "sources": sources,
            "enhanced_query": enhanced_query,
            "namespace": namespace, "session_id": session_id,
            "smart_info": smart_info,
            "run_id": run_id,
        }

        # ── Stream tokens ───────────────────────────────────────────
        full_answer_parts = []
        for token in self.generator.generate_stream(
            query=user_query, documents=final_docs, namespace=pinecone_namespace,
            chat_history=chat_history, session_id=session_id, enhanced_query=enhanced_query,
        ):
            full_answer_parts.append(token)
            yield {"type": "token", "content": token}

        full_answer = "".join(full_answer_parts)
        if session_id:
            self.memory.add_turn(session_id, user_query, full_answer)

        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info("⏱ TOTAL: %.2fs", latency_ms / 1000)


# ═════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═════════════════════════════════════════════════════════════════════════════

_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
