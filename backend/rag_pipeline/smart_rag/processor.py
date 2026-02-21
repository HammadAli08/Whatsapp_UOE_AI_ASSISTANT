"""
Smart RAG Processor — Self-Correcting Retrieval with Best-Effort Answering

Coordinates the grade → retry → rewrite loop using the SmartChunkGrader
and SmartQueryRewriter. Produces metrics and fallback messages.

Features:
  - Up to 6 retry attempts with progressive strategy escalation
  - Best-effort answering: after exhausting all retries, collects ALL
    relevant chunks found across every attempt and uses them for generation
  - Clarification detection: when results are thin after all retries,
    generates specific follow-up questions for the user
  - Only uses the "sorry" fallback when literally zero chunks exist
"""

import json
import logging
from typing import Dict, List, Optional

from openai import OpenAI

from langsmith import traceable

from ..config import OPENAI_API_KEY
from .config import SMART_RAG_CONFIG, NO_RESULTS_MESSAGE, CLARIFICATION_MESSAGE_TEMPLATE
from .grader import SmartChunkGrader
from .rewriter import SmartQueryRewriter

logger = logging.getLogger(__name__)


class SmartRAGProcessor:
    """
    Self-correcting retrieval processor with best-effort answering.

    Flow per attempt:
      1. Retrieve top-k chunks (handled by the pipeline)
      2. Grade each chunk: relevant (with confidence) / irrelevant (with reason)
      3. If relevant >= min_relevant_chunks → proceed to generation
      4. Else → rewrite the query using reasons from irrelevant chunks → retry
      5. After 6 retries: answer with ALL relevant chunks collected so far
      6. If few chunks with low confidence → ask user for clarification
      7. If zero chunks across all attempts → return fallback message
    """

    def __init__(self):
        self.grader = SmartChunkGrader()
        self.rewriter = SmartQueryRewriter()
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.min_relevant = SMART_RAG_CONFIG["min_relevant_chunks"]
        self.max_retries = SMART_RAG_CONFIG["max_retries"]
        self.early_success = SMART_RAG_CONFIG.get("early_success_threshold", 4)

    def grade_chunks(self, query: str, chunks: List[Dict]):
        """Delegate to SmartChunkGrader."""
        return self.grader.grade_chunks(query, chunks)

    def rewrite_query(self, original_query: str, failed_chunks: List[Dict], attempt: int) -> str:
        """Delegate to SmartQueryRewriter."""
        return self.rewriter.rewrite(original_query, failed_chunks, attempt)

    def should_retry(self, relevant_chunks: List[Dict], attempt: int) -> bool:
        """Return True if we have too few relevant chunks and can still retry."""
        # Early success: if we found enough high-quality chunks, stop
        if len(relevant_chunks) >= self.early_success:
            return False
        return len(relevant_chunks) < self.min_relevant and attempt < self.max_retries

    def should_stop_early(self, all_relevant: List[Dict], attempt: int) -> bool:
        """
        Stop early if we've accumulated enough good chunks across iterations.
        Even if this particular attempt didn't meet min_relevant, the total
        across all attempts might be sufficient.
        """
        return len(all_relevant) >= self.early_success

    @traceable(name="smart_rag.detect_clarification", run_type="chain")
    def detect_clarification_needed(
        self,
        user_query: str,
        all_relevant: List[Dict],
        all_irrelevant: List[Dict],
    ) -> Optional[str]:
        """
        Use LLM to decide if the user needs to provide more details.

        Called after exhausting all retries when we have very few relevant chunks.
        Returns a clarification message with specific follow-up suggestions,
        or None if clarification isn't helpful.
        """
        # Only trigger clarification when we have 0 or very few results
        if len(all_relevant) >= self.min_relevant:
            return None

        try:
            # Build context about what was found / not found
            found_summary = "Nothing relevant found." if not all_relevant else "\n".join(
                f"- {c.get('metadata', {}).get('source_file', 'Unknown')}: "
                f"{c.get('text', '')[:100]}..."
                for c in all_relevant[:3]
            )

            irrelevant_reasons = "\n".join(
                f"- {c.get('grade_reason', 'N/A')}"
                for c in all_irrelevant[:5]
            )

            prompt = (
                f"A user asked: \"{user_query}\"\n\n"
                f"After 7 retrieval attempts in a university document system "
                f"(University of Education, Lahore), here's what we found:\n\n"
                f"Relevant results ({len(all_relevant)} chunks):\n{found_summary}\n\n"
                f"Common rejection reasons:\n{irrelevant_reasons}\n\n"
                f"Based on the query and what was found, suggest 2-3 SPECIFIC "
                f"follow-up questions or details the user could provide to help "
                f"find better results. Focus on:\n"
                f"- Missing specifics (program name, department, batch year, semester)\n"
                f"- Category confusion (BS/ADP vs MS/PhD vs Rules)\n"
                f"- Ambiguous terms that could mean different things\n\n"
                f"Return ONLY the bullet-point suggestions, nothing else. "
                f"Keep each suggestion under 20 words."
            )

            resp = self.client.chat.completions.create(
                model=SMART_RAG_CONFIG["clarification_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )
            suggestions = resp.choices[0].message.content.strip()

            if suggestions:
                return CLARIFICATION_MESSAGE_TEMPLATE.format(suggestions=suggestions)

        except Exception as exc:
            logger.warning("Clarification detection failed: %s", exc)

        return None

    @staticmethod
    def build_metrics(
        *,
        total_retrievals: int,
        total_chunks_graded: int,
        query_rewrites: List[Dict],
        final_relevant_count: int,
        used_fallback: bool,
        best_effort: bool,
        clarification_asked: bool = False,
    ) -> Dict:
        return {
            "total_retrievals": total_retrievals,
            "total_chunks_graded": total_chunks_graded,
            "query_rewrites": query_rewrites,
            "final_relevant_chunks": final_relevant_count,
            "used_fallback": used_fallback,
            "best_effort": best_effort,
            "clarification_asked": clarification_asked,
        }


def get_fallback_message() -> str:
    """Graceful fallback when no chunks survive all retries."""
    return NO_RESULTS_MESSAGE


# ═════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═════════════════════════════════════════════════════════════════════════════

_processor: Optional[SmartRAGProcessor] = None


def get_smart_processor() -> SmartRAGProcessor:
    """Get or create SmartRAGProcessor singleton."""
    global _processor
    if _processor is None:
        _processor = SmartRAGProcessor()
    return _processor
