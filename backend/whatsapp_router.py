"""
WhatsApp Webhook Router

Receives incoming WhatsApp messages via the Meta Cloud API webhook,
routes them through the RAG pipeline, and sends the response back.

CRITICAL: Meta expects a 200 OK within ~5 seconds.  All heavy work
(RAG query, reply send) runs as a background task so the webhook
responds instantly.
"""

import os
import asyncio
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Request, Response, BackgroundTasks

from rag_pipeline import get_pipeline
from rag_pipeline.config import VALID_NAMESPACES

logger = logging.getLogger(__name__)

# ── WhatsApp Cloud API credentials ──────────────────────────────────────────
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
WHATSAPP_DEFAULT_NAMESPACE = os.getenv("WHATSAPP_DEFAULT_NAMESPACE", "bs-adp")
WHATSAPP_API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v22.0")

# ── Router ──────────────────────────────────────────────────────────────────
whatsapp_router = APIRouter(prefix="/api/whatsapp", tags=["WhatsApp"])


# ═════════════════════════════════════════════════════════════════════════════
# WEBHOOK ENDPOINT  (GET = Meta verification, POST = incoming messages)
# ═════════════════════════════════════════════════════════════════════════════

@whatsapp_router.api_route("/webhook", methods=["GET", "POST"])
async def whatsapp_webhook(request: Request, bg: BackgroundTasks) -> Response:
    """
    GET  → Meta webhook verification handshake.
    POST → Incoming user messages from WhatsApp Cloud API.

    The POST handler returns 200 OK *immediately* and schedules
    all heavy processing (RAG query + reply) as a background task.
    """

    # ── GET: Webhook verification ────────────────────────────────────────
    if request.method == "GET":
        params = request.query_params
        mode = params.get("hub.mode")
        token = params.get("hub.verify_token")
        challenge = params.get("hub.challenge")

        if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
            logger.info("WhatsApp webhook verified successfully")
            return Response(content=challenge, status_code=200)

        logger.warning("WhatsApp webhook verification failed — token mismatch")
        return Response(content="Verification failed", status_code=403)

    # ── POST: Incoming message ───────────────────────────────────────────
    try:
        body = await request.json()

        entries = body.get("entry", [])
        if not entries:
            return Response(content="No entry", status_code=200)

        for entry in entries:
            changes = entry.get("changes", [])
            for change in changes:
                value = change.get("value", {})

                # Skip status updates (delivered, read, etc.)
                if "statuses" in value:
                    continue

                messages = value.get("messages", [])
                if not messages:
                    continue

                message = messages[0]
                from_number = message.get("from", "")
                msg_type = message.get("type", "")

                # ── Extract text from message ────────────────────────
                user_text = _extract_text(message, msg_type)
                if not user_text:
                    logger.info(
                        "Unsupported message type '%s' from %s — skipping",
                        msg_type, from_number,
                    )
                    # Fire-and-forget: tell the user we only handle text
                    bg.add_task(
                        _send_text,
                        from_number,
                        "⚠️ Sorry, I can only process text messages at the moment. "
                        "Please send your question as text.",
                    )
                    return Response(content="OK", status_code=200)

                # ── Namespace detection (optional prefix) ────────────
                namespace, clean_query = _parse_namespace_prefix(user_text)

                logger.info(
                    "WhatsApp → from=%s namespace=%s query='%s'",
                    from_number, namespace, clean_query[:80],
                )

                # ── Schedule heavy work in background ────────────────
                bg.add_task(
                    _process_and_reply,
                    from_number,
                    clean_query,
                    namespace,
                )

        # Return 200 OK immediately so Meta doesn't timeout
        return Response(content="OK", status_code=200)

    except Exception as exc:
        logger.error("WhatsApp webhook error: %s", exc, exc_info=True)
        return Response(content="OK", status_code=200)


# ═════════════════════════════════════════════════════════════════════════════
# BACKGROUND WORKER
# ═════════════════════════════════════════════════════════════════════════════

def _process_and_reply(from_number: str, query: str, namespace: str) -> None:
    """
    Run the RAG pipeline synchronously (it's CPU/IO-bound) and send
    the reply back to the user.  This runs in a background thread
    managed by FastAPI's BackgroundTasks.
    """
    try:
        pipeline = get_pipeline()
        result = pipeline.query(
            user_query=query,
            namespace=namespace,
            enhance_query=True,
            session_id=from_number,  # phone number = session ID
            enable_smart=False,
        )
        reply = _format_reply(result)
    except Exception as pipe_err:
        logger.error("Pipeline error: %s", pipe_err, exc_info=True)
        reply = (
            "❌ I encountered an error processing your question. "
            "Please try again in a moment."
        )

    # Send reply (sync wrapper around async httpx)
    try:
        asyncio.run(_send_text(from_number, reply))
    except Exception as send_err:
        logger.error("Failed to send WhatsApp reply: %s", send_err, exc_info=True)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _extract_text(message: dict, msg_type: str) -> Optional[str]:
    """Extract plain text from a WhatsApp message object."""
    if msg_type == "text":
        return message.get("text", {}).get("body", "").strip()
    # Interactive list/button replies
    if msg_type == "interactive":
        interactive = message.get("interactive", {})
        reply = interactive.get("button_reply") or interactive.get("list_reply")
        if reply:
            return reply.get("title", "").strip()
    return None


def _parse_namespace_prefix(text: str) -> tuple[str, str]:
    """
    Allow users to switch namespace with a prefix, e.g.:
        /rules What is the attendance policy?
        /ms-phd PhD admission requirements

    Falls back to the default namespace if no prefix is found.
    """
    if text.startswith("/"):
        parts = text.split(None, 1)
        prefix = parts[0][1:].lower()  # remove the leading /
        if prefix in VALID_NAMESPACES and len(parts) > 1:
            return prefix, parts[1].strip()
    return WHATSAPP_DEFAULT_NAMESPACE, text


def _format_reply(result: dict) -> str:
    """Format the RAG pipeline result into a WhatsApp-friendly message."""
    answer = result.get("answer", "No answer generated.")

    # Append source references if available
    sources = result.get("sources", [])
    if sources:
        unique_files = []
        seen = set()
        for src in sources[:3]:  # top 3 sources max
            fname = src.get("file", "")
            if fname and fname not in seen:
                seen.add(fname)
                page = src.get("page", "")
                label = f"📄 {fname}"
                if page and page != "N/A":
                    label += f" (p. {page})"
                unique_files.append(label)

        if unique_files:
            answer += "\n\n─── Sources ───\n" + "\n".join(unique_files)

    return answer


async def _send_text(to: str, text: str) -> bool:
    """Send a text message via the WhatsApp Cloud API."""
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        logger.error("WhatsApp credentials not configured (WHATSAPP_TOKEN / WHATSAPP_PHONE_NUMBER_ID)")
        return False

    url = (
        f"https://graph.facebook.com/{WHATSAPP_API_VERSION}"
        f"/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    )
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    # WhatsApp has a 4096 character limit per text message
    if len(text) > 4000:
        text = text[:3990] + "\n\n…(truncated)"

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": text},
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=payload)

        if resp.status_code == 200:
            logger.info("WhatsApp message sent to %s", to)
            return True

        logger.error(
            "WhatsApp API error: status=%d body=%s",
            resp.status_code, resp.text[:500],
        )
        return False

    except httpx.HTTPError as http_err:
        logger.error("WhatsApp HTTP error: %s", http_err)
        return False
