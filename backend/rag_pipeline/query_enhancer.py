"""
Query Enhancement Module

Uses GPT-4o-mini to rewrite user queries for optimal vector database search.
GPT-4o-mini is used instead of GPT-4o for speed — this is a lightweight
rewrite task that needs to be fast, not deeply reasoned.

Now conversation-aware: passes last 2 turns of chat history so follow-up
queries like "What about its outline?" resolve pronouns correctly.
"""

import logging
from typing import List, Dict, Optional
from openai import OpenAI

from langsmith import traceable

from .config import (
    OPENAI_API_KEY,
    QUERY_ENHANCER_TIMEOUT_SECONDS,
    SYSTEM_PROMPTS_DIR,
    QUERY_ENHANCER_PROMPT_FILE,
)

logger = logging.getLogger(__name__)

# Fast model for query rewriting — GPT-4o-mini has ~2x lower latency than GPT-4o
_ENHANCER_MODEL = "gpt-4o-mini"
_ENHANCER_MAX_TOKENS = 64  # bumped from 48 to handle resolved follow-ups
_ENHANCER_TEMPERATURE = 0.0  # deterministic
_MAX_HISTORY_MESSAGES = 4  # last 2 turns (2 user + 2 assistant)


class QueryEnhancer:
    """
    Enhances user queries for optimal vector search using GPT-4o-mini.
    Optionally uses recent chat history to resolve pronouns and carry
    forward context from previous turns.
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self._prompt: Optional[str] = None

    @property
    def prompt(self) -> str:
        """Lazy load the query enhancer system prompt."""
        if self._prompt is None:
            prompt_path = SYSTEM_PROMPTS_DIR / QUERY_ENHANCER_PROMPT_FILE
            if prompt_path.exists():
                self._prompt = prompt_path.read_text().strip()
            else:
                self._prompt = (
                    "Rewrite the user's query for optimal vector database search. "
                    "Keep it SHORT (under 20 words). Extract key academic terms. "
                    "Remove filler words. Output ONLY the enhanced query."
                )
        return self._prompt

    @staticmethod
    def _build_context_block(chat_history: List[Dict[str, str]]) -> str:
        """Format the last few turns into a compact context string."""
        recent = chat_history[-_MAX_HISTORY_MESSAGES:]
        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            # Truncate long assistant answers to save tokens
            content = msg["content"]
            if role == "Assistant" and len(content) > 150:
                content = content[:150] + "…"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @traceable(name="query_enhancer.enhance", run_type="chain")
    def enhance(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Enhance the query for better vector search.

        Uses GPT-4o-mini with minimal tokens and temperature=0
        for speed + determinism. Typical latency: 200-400ms.

        If chat_history is provided, injects the last 2 turns so the
        model can resolve pronouns and carry forward context.

        Falls back to original query on any failure.
        """
        if not query or not query.strip():
            return query

        normalized_query = query.strip()
        if len(normalized_query.split()) <= 4 and not chat_history:
            return normalized_query

        # Build the user message — optionally with conversation context
        if chat_history:
            context_block = self._build_context_block(chat_history)
            user_content = (
                f"CONVERSATION CONTEXT:\n{context_block}\n\n"
                f"CURRENT QUERY: {normalized_query}"
            )
        else:
            user_content = normalized_query

        try:
            response = self.client.chat.completions.create(
                model=_ENHANCER_MODEL,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=_ENHANCER_TEMPERATURE,
                max_tokens=_ENHANCER_MAX_TOKENS,
                timeout=QUERY_ENHANCER_TIMEOUT_SECONDS,
            )

            enhanced = response.choices[0].message.content
            if enhanced and enhanced.strip():
                return enhanced.strip()

            return normalized_query

        except Exception as exc:
            logger.debug("Query enhancement failed (using original): %s", exc)
            return normalized_query


# Singleton instance
_query_enhancer: Optional[QueryEnhancer] = None


def get_query_enhancer() -> QueryEnhancer:
    """Get or create QueryEnhancer singleton."""
    global _query_enhancer
    if _query_enhancer is None:
        _query_enhancer = QueryEnhancer()
    return _query_enhancer
