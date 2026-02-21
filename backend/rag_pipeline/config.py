"""
RAG Pipeline Configuration

Central configuration for all RAG pipeline components.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API KEYS
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# =============================================================================
# LANGSMITH TRACING
# =============================================================================

_tracing_raw = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or os.getenv("LANGCHAIN_TRACING") or "false"
LANGSMITH_TRACING_ENABLED = _tracing_raw.lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "uoe-ai-assistant")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# Ensure the SDK env vars are set so LangSmith auto-instruments OpenAI calls
if LANGSMITH_TRACING_ENABLED and LANGSMITH_API_KEY:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT

# =============================================================================
# REDIS CONFIGURATION (Short-Term Memory — Redis Cloud)
# =============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "default")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "10"))
MEMORY_TTL_SECONDS = int(os.getenv("MEMORY_TTL_SECONDS", "1800"))

# =============================================================================
# PINECONE CONFIGURATION
# =============================================================================

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "uoeaiassistant")

NAMESPACE_MAP = {
    "bs-adp": "bs-adp-schemes",
    "ms-phd": "ms-phd-schemes",
    "rules": "rules-regulations",
}

VALID_NAMESPACES = list(NAMESPACE_MAP.keys())

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPTS_DIR = Path(__file__).parent.parent / "system_prompts"

SYSTEM_PROMPT_FILES = {
    "bs-adp-schemes": "bs_adp_systemprompt.txt",
    "ms-phd-schemes": "ms_phd_systemprompt.txt",
    "rules-regulations": "rules&regulations.txt",
}

QUERY_ENHANCER_PROMPT_FILE = "query_enhancer_prompt.txt"
SMART_GRADING_PROMPT_FILE = "smart_grading_prompt.txt"
SMART_REWRITE_PROMPT_FILE = "smart_rewrite_prompt.txt"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "3072"))
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_CHAT_TEMPERATURE = float(os.getenv("OPENAI_CHAT_TEMPERATURE", "0.1"))
OPENAI_CHAT_MAX_TOKENS = int(os.getenv("OPENAI_CHAT_MAX_TOKENS", "1500"))

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================

DEFAULT_TOP_K_RETRIEVE = int(os.getenv("DEFAULT_TOP_K_RETRIEVE", "5"))
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "512"))
EMBEDDING_CACHE_TTL_SECONDS = float(os.getenv("EMBEDDING_CACHE_TTL_SECONDS", "3600"))
RETRIEVAL_CACHE_TTL_SECONDS = float(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS", "45"))

# =============================================================================
# TIMEOUTS
# =============================================================================

QUERY_ENHANCER_TIMEOUT_SECONDS = float(os.getenv("QUERY_ENHANCER_TIMEOUT_SECONDS", "1.5"))

# =============================================================================
# SMART RAG CONFIGURATION
# =============================================================================

# Model used for chunk grading and query rewriting (cheap + fast)
SMART_RAG_CLASSIFIER_MODEL = "gpt-4o-mini"

# =============================================================================
# FEEDBACK CONFIGURATION
# =============================================================================

FEEDBACK_LOG_PATH = Path(__file__).parent.parent / "feedback_log.jsonl"
