"""
Short-Term Conversation Memory Module

Redis Cloud-backed per-session chat history with TTL-based expiry.
"""

import json
import uuid
import logging
from typing import List, Dict, Optional

import redis

from .config import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_USERNAME,
    REDIS_PASSWORD,
    MEMORY_MAX_TURNS,
    MEMORY_TTL_SECONDS,
)

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Redis Cloud-backed short-term conversation memory."""

    KEY_PREFIX = "uoe:memory:"

    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self.max_turns = MEMORY_MAX_TURNS
        self.ttl = MEMORY_TTL_SECONDS

    def connect(self) -> None:
        """Establish Redis Cloud connection."""
        try:
            self._client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                username=REDIS_USERNAME,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            self._client.ping()
            logger.info("Redis Cloud connected (%s:%s)", REDIS_HOST, REDIS_PORT)
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            logger.warning("Redis Cloud unavailable: %s", exc)
            self._client = None

    def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._client:
            self._client.close()
            logger.info("Redis disconnected")

    @property
    def available(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError):
            return False

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        if not self.available:
            return []
        key = f"{self.KEY_PREFIX}{session_id}"
        try:
            raw = self._client.get(key)
            if raw is None:
                return []
            return json.loads(raw)
        except (redis.RedisError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read memory for %s: %s", session_id, exc)
            return []

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        if not self.available:
            return
        key = f"{self.KEY_PREFIX}{session_id}"
        try:
            history = self.get_history(session_id)
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": assistant_message})
            max_messages = self.max_turns * 2
            if len(history) > max_messages:
                history = history[-max_messages:]
            self._client.set(key, json.dumps(history), ex=self.ttl)
        except redis.RedisError as exc:
            logger.warning("Failed to write memory for %s: %s", session_id, exc)

    def clear(self, session_id: str) -> bool:
        if not self.available:
            return False
        key = f"{self.KEY_PREFIX}{session_id}"
        try:
            return bool(self._client.delete(key))
        except redis.RedisError as exc:
            logger.warning("Failed to clear memory for %s: %s", session_id, exc)
            return False

    def get_session_info(self, session_id: str) -> Dict:
        if not self.available:
            return {"exists": False, "turns": 0, "ttl_remaining": 0}
        key = f"{self.KEY_PREFIX}{session_id}"
        try:
            raw = self._client.get(key)
            ttl = self._client.ttl(key)
            if raw is None:
                return {"exists": False, "turns": 0, "ttl_remaining": 0}
            messages = json.loads(raw)
            return {"exists": True, "turns": len(messages) // 2, "ttl_remaining": max(ttl, 0)}
        except (redis.RedisError, json.JSONDecodeError):
            return {"exists": False, "turns": 0, "ttl_remaining": 0}


_memory: Optional[ConversationMemory] = None


def get_memory() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
        _memory.connect()
    return _memory
