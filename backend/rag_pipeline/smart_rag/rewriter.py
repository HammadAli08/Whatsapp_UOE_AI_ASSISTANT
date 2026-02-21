"""
Smart Query Rewriter — 3-Attempt Strategy Escalation

Rewrites failed queries using context from irrelevant chunks (including reasons
they were rejected), with progressive strategy escalation across 3 attempts:
  - Attempt 1: Add specific keywords (course codes, full program names)
  - Attempt 2: Add metadata hints (batch year, semester, program type)
  - Attempt 3: Broaden / generalize the query as a last resort
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

from langsmith import traceable

from ..config import OPENAI_API_KEY, SYSTEM_PROMPTS_DIR
from .config import SMART_RAG_CONFIG

logger = logging.getLogger(__name__)

_REWRITE_PROMPT_FILE = "smart_rewrite_prompt.txt"


class SmartQueryRewriter:
    """
    Rewrites queries that led to irrelevant retrievals.

    Uses attempt-aware strategy escalation (3 levels) and includes
    reasons why previous chunks were irrelevant to guide the rewrite.
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = SMART_RAG_CONFIG["rewriting_model"]
        self._prompt_template: Optional[str] = None

    @property
    def prompt_template(self) -> str:
        """Lazy-load the rewrite prompt from system_prompts/."""
        if self._prompt_template is None:
            prompt_path = SYSTEM_PROMPTS_DIR / _REWRITE_PROMPT_FILE
            if prompt_path.exists():
                self._prompt_template = prompt_path.read_text().strip()
            else:
                self._prompt_template = (
                    "Original Query: {original_query}\n"
                    "Attempt Number: {attempt}\n"
                    "Failed chunks: {failed_chunks_summary}\n"
                    "Reasons: {failed_reasons}\n\n"
                    "Rewrite the query to be more specific. "
                    "Return ONLY the rewritten query.\n\n"
                    "Rewritten Query:"
                )
        return self._prompt_template

    @traceable(name="smart_rag.rewrite_query", run_type="chain")
    def rewrite(
        self,
        original_query: str,
        failed_chunks: List[Dict],
        attempt: int,
    ) -> str:
        """
        Rewrite the query using context from failed chunks.

        Args:
            original_query: The user's original question
            failed_chunks:  Chunks deemed irrelevant (may have 'grade_reason')
            attempt:        Current attempt number (1, 2, 3)

        Returns:
            A rewritten query string
        """
        try:
            # Build failed chunk summaries
            summaries = []
            reasons = []
            for chunk in failed_chunks[:5]:
                m = chunk.get("metadata", {})
                summaries.append(
                    f"- Source: {m.get('source_file', 'Unknown')}, "
                    f"Dept: {m.get('department', 'N/A')}, "
                    f"Program: {m.get('program_type', 'N/A')}"
                )
                reason = chunk.get("grade_reason", "Not specified")
                reasons.append(f"- {reason}")

            failed_summary = "\n".join(summaries) if summaries else "No context available"
            failed_reasons = "\n".join(reasons) if reasons else "No reasons available"

            prompt = self.prompt_template.format(
                original_query=original_query,
                attempt=attempt,
                failed_chunks_summary=failed_summary,
                failed_reasons=failed_reasons,
            )

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100,
            )
            rewritten = resp.choices[0].message.content.strip()

            logger.info(
                "Smart rewrite (attempt %d): '%s' → '%s'",
                attempt, original_query, rewritten,
            )
            return rewritten
        except Exception as exc:
            logger.warning("Smart query rewriting failed: %s — keeping original", exc)
            return original_query
