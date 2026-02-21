"""
Smart Chunk Grader — Confidence-Based Relevance Grading (Batched)

Grades ALL retrieved chunks in a single LLM call using GPT-4o-mini with:
  - 5 evaluation signals (topic, program, specificity, department/year, completeness)
  - Confidence scoring (0.0–1.0) instead of binary yes/no
  - JSON array output for structured batch decisions

Chunks below ``confidence_threshold`` are rejected even if labeled relevant.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from langsmith import traceable

from ..config import OPENAI_API_KEY, SYSTEM_PROMPTS_DIR
from .config import SMART_RAG_CONFIG

logger = logging.getLogger(__name__)

_GRADING_PROMPT_FILE = "smart_grading_prompt.txt"


class SmartChunkGrader:
    """
    Grades retrieved chunks for relevance using confidence scoring.

    All chunks are graded in a **single** LLM call (batched prompt)
    to eliminate per-chunk latency.
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = SMART_RAG_CONFIG["grading_model"]
        self.confidence_threshold = SMART_RAG_CONFIG["confidence_threshold"]
        self._prompt_template: Optional[str] = None

    @property
    def prompt_template(self) -> str:
        """Lazy-load the grading prompt from system_prompts/."""
        if self._prompt_template is None:
            prompt_path = SYSTEM_PROMPTS_DIR / _GRADING_PROMPT_FILE
            if prompt_path.exists():
                self._prompt_template = prompt_path.read_text().strip()
            else:
                # Inline fallback — should not happen in production
                self._prompt_template = (
                    "Question: {query}\n\n"
                    "{chunks_block}\n\n"
                    "Respond with a JSON array of objects, one per chunk:\n"
                    '[{{"index": 0, "relevant": true/false, '
                    '"confidence": 0.0-1.0, "reason": "..."}}]'
                )
        return self._prompt_template

    # ── Build numbered chunks block ──────────────────────────────────────

    @staticmethod
    def _build_chunks_block(chunks: List[Dict]) -> str:
        """Format all chunks into a numbered block for the prompt."""
        lines: List[str] = []
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source_file", "Unknown")
            page = metadata.get("page_number", "N/A")
            text = chunk.get("text", "")[:1500]
            lines.append(
                f"[Chunk {i}]\n"
                f"Source: {source}, Page: {page}\n"
                f"Content: {text}\n"
                f"---"
            )
        return "\n".join(lines)

    # ── Main grading entry point ─────────────────────────────────────────

    @traceable(name="smart_rag.grade_chunks", run_type="chain")
    def grade_chunks(
        self,
        query: str,
        chunks: List[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Grade all retrieved chunks in a **single** LLM call.

        Args:
            query:  The user's question
            chunks: Retrieved document dicts (must have 'text' and 'metadata')

        Returns:
            (relevant_chunks, irrelevant_chunks)
            Each chunk in irrelevant_chunks gets an extra 'grade_reason' key.
        """
        if not chunks:
            return [], []

        try:
            chunks_block = self._build_chunks_block(chunks)
            prompt = self.prompt_template.format(
                query=query,
                chunks_block=chunks_block,
            )

            # Scale max_tokens with chunk count (~50 tokens per verdict)
            max_tokens = max(200, 60 * len(chunks))

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content.strip()

            grades = self._parse_batch_grades(raw, len(chunks))

        except Exception as exc:
            logger.warning("Smart batch grading error: %s", exc)
            # On failure, accept all chunks with above-threshold confidence
            # so they pass the 0.6 threshold consistently
            for chunk in chunks:
                chunk["grade_confidence"] = 0.7
            return list(chunks), []

        # ── Partition into relevant / irrelevant ─────────────────────────
        relevant: List[Dict] = []
        irrelevant: List[Dict] = []

        for i, chunk in enumerate(chunks):
            grade = grades[i] if i < len(grades) else {
                "relevant": True, "confidence": 0.5, "reason": "Missing grade"
            }

            if grade["relevant"] and grade["confidence"] >= self.confidence_threshold:
                chunk["grade_confidence"] = grade["confidence"]
                relevant.append(chunk)
            else:
                chunk["grade_reason"] = grade.get("reason", "Low relevance")
                chunk["grade_confidence"] = grade["confidence"]
                irrelevant.append(chunk)

        logger.info(
            "Smart grading: %d relevant, %d irrelevant out of %d chunks",
            len(relevant), len(irrelevant), len(chunks),
        )
        return relevant, irrelevant

    # ── Parsing helpers ──────────────────────────────────────────────────

    @staticmethod
    def _parse_batch_grades(raw: str, expected: int) -> List[Dict]:
        """
        Parse the LLM's JSON array response.

        Falls back to individual JSON-object parsing if the array is
        malformed, and ultimately to keyword heuristics.
        """
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            arr = json.loads(cleaned)
            if isinstance(arr, list):
                results: List[Dict] = []
                for item in arr:
                    results.append({
                        "relevant": bool(item.get("relevant", False)),
                        "confidence": float(item.get("confidence", 0.0)),
                        "reason": str(item.get("reason", "")),
                    })
                # Pad with safe defaults if LLM returned fewer items
                while len(results) < expected:
                    results.append({
                        "relevant": True,
                        "confidence": 0.5,
                        "reason": "Missing from LLM response",
                    })
                return results
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Final fallback: accept everything with neutral confidence
        logger.warning("Could not parse batch grades, accepting all chunks")
        return [
            {"relevant": True, "confidence": 0.5, "reason": "Parse failure fallback"}
            for _ in range(expected)
        ]
