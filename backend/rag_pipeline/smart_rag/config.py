"""
Smart RAG Configuration

Constants for the self-correcting retrieval system with best-effort answering.

Features:
  - 6 retry attempts with progressive strategy escalation
  - Best-effort answering: after exhausting retries, answers with whatever
    chunks were collected rather than returning empty
  - Clarification detection: when results are thin, asks user for details
  - Only uses fallback "sorry" message when literally zero chunks exist
"""

SMART_RAG_CONFIG = {
    "max_retries": 6,                # Max re-retrieval attempts after initial try
    "min_relevant_chunks": 2,        # Minimum relevant chunks needed to skip retries
    "confidence_threshold": 0.6,     # Minimum confidence score for a chunk to be relevant
    "grading_model": "gpt-4o-mini",  # Model for grading chunks (cheap + fast)
    "rewriting_model": "gpt-4o-mini",# Model for query rewriting
    "retry_top_k_boost": 4,          # Extra chunks to retrieve on each retry
    "clarification_model": "gpt-4o-mini",  # Model for clarification detection
    "early_success_threshold": 4,    # If this many relevant chunks found, stop immediately
}

NO_RESULTS_MESSAGE = (
    "I'm sorry, I wasn't able to find relevant information to answer your "
    "question after multiple retrieval attempts.\n\n"
    "Suggestions:\n"
    "- Try selecting a different category "
    "(BS/ADP, MS/PhD, or Rules & Regulations)\n"
    "- Add more specific details to your question "
    "(course code, program name, batch year)\n"
    "- Rephrase your question using different keywords\n\n"
    "The information you're looking for might not be in the current "
    "document collection."
)

CLARIFICATION_MESSAGE_TEMPLATE = (
    "I found some partial information but I'm not fully confident in the results. "
    "Could you help me narrow down the search?\n\n"
    "{suggestions}\n\n"
    "With more details, I can find more accurate information for you."
)
