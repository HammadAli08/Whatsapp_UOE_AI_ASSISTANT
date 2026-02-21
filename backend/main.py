"""
FastAPI Backend for UOE AI Assistant

Provides REST API endpoints for the RAG pipeline with streaming support.
"""

import os
import json
import uuid
import signal
import socket
import logging
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langsmith import Client as LangSmithClient

from rag_pipeline import get_pipeline
from rag_pipeline.config import (
    DEFAULT_TOP_K_RETRIEVE,
    LANGSMITH_TRACING_ENABLED,
    LANGSMITH_PROJECT,
    LANGSMITH_API_KEY,
    FEEDBACK_LOG_PATH,
)
from whatsapp_router import whatsapp_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Silence noisy loggers ──────────────────────────────────
# Health-check requests flood the log; suppress uvicorn.access for those.
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _port_is_free(port: int) -> bool:
    """Return True if nothing is listening on *port*."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _free_port(port: int) -> None:
    """Kill any process occupying *port* so the server can bind."""
    if _port_is_free(port):
        return
    logger.warning("Port %d is in use — attempting to free it…", port)
    try:
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            try:
                pid_int = int(pid)
                logger.info("Killing stale process %d on port %d", pid_int, port)
                import os as _os
                _os.kill(pid_int, signal.SIGKILL)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        import time as _time
        _time.sleep(0.5)
    except Exception as exc:
        logger.warning("Could not free port %d: %s", port, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting UOE AI Assistant API...")
    _ = get_pipeline()
    logger.info("Pipeline initialized successfully")
    if LANGSMITH_TRACING_ENABLED:
        logger.info("LangSmith tracing ENABLED → project: %s", LANGSMITH_PROJECT)
    else:
        logger.warning("LangSmith tracing DISABLED (check LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY)")
    yield
    logger.info("Shutting down UOE AI Assistant API...")


app = FastAPI(
    title="UOE AI Assistant API",
    description="RAG-based AI assistant for University of Education queries",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WhatsApp Webhook Router ─────────────────────────────────────────────────
app.include_router(whatsapp_router)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    namespace: str = Field(default="bs-adp")
    enhance_query: bool = Field(default=True)
    enable_smart: bool = Field(default=False, description="Enable Smart RAG with best-effort self-correcting retrieval (6 retries, answers with best available chunks)")
    top_k_retrieve: int = Field(default=DEFAULT_TOP_K_RETRIEVE, ge=1, le=20)
    session_id: Optional[str] = Field(default=None)


class ChatResponse(BaseModel):
    answer: str
    sources: list
    enhanced_query: str
    namespace: str
    session_id: str
    run_id: Optional[str] = Field(default=None, description="LangSmith trace run ID for feedback linkage")
    smart_info: Optional[dict] = Field(default=None, description="Smart RAG metrics: total_retrievals, query_rewrites, final_relevant_chunks, best_effort, etc.")


class FeedbackRequest(BaseModel):
    run_id: str = Field(..., min_length=1, description="LangSmith run ID to attach feedback to")
    score: int = Field(..., ge=0, le=1, description="1 for thumbs up, 0 for thumbs down")
    comment: Optional[str] = Field(default=None, max_length=1000)


@app.get("/")
async def root():
    return {"message": "UOE AI Assistant API", "status": "running", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health probe — intentionally silent in logs."""
    return {"status": "healthy"}


@app.middleware("http")
async def _access_log_middleware(request: Request, call_next):
    """Log requests except high-frequency health probes."""
    response = await call_next(request)
    if request.url.path not in ("/health", "/"):
        logger.info("%s %s → %s", request.method, request.url.path, response.status_code)
    return response


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    try:
        pipeline = get_pipeline()
        session_id = request.session_id or str(uuid.uuid4())

        result = pipeline.query(
            user_query=request.query,
            namespace=request.namespace,
            enhance_query=request.enhance_query,
            top_k_retrieve=request.top_k_retrieve,
            session_id=session_id,
            enable_smart=request.enable_smart,
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            enhanced_query=result["enhanced_query"],
            namespace=result["namespace"],
            session_id=session_id,
            run_id=result.get("run_id"),
            smart_info=result.get("smart_info"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    try:
        pipeline = get_pipeline()
        session_id = request.session_id or str(uuid.uuid4())

        def event_generator():
            try:
                for chunk in pipeline.stream_query(
                    user_query=request.query,
                    namespace=request.namespace,
                    enhance_query=request.enhance_query,
                    top_k_retrieve=request.top_k_retrieve,
                    session_id=session_id,
                    enable_smart=request.enable_smart,
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Stream setup error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/namespaces")
async def get_namespaces():
    """Get available namespaces."""
    return {
        "namespaces": [
            {"id": "bs-adp", "name": "BS / ADP Programs", "description": "Bachelor's and Associate Degree Programs"},
            {"id": "ms-phd", "name": "MS / PhD Programs", "description": "Master's and Doctoral Programs"},
            {"id": "rules", "name": "Rules & Regulations", "description": "University policies and regulations"},
        ]
    }


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit thumbs-up/down feedback linked to a LangSmith trace."""
    try:
        # ── 1. Send to LangSmith ─────────────────────────────────────
        if LANGSMITH_TRACING_ENABLED and LANGSMITH_API_KEY:
            try:
                ls_client = LangSmithClient()
                ls_client.create_feedback(
                    run_id=request.run_id,
                    key="user-score",
                    score=request.score,
                    comment=request.comment,
                )
                logger.info(
                    "Feedback sent to LangSmith: run_id=%s score=%d",
                    request.run_id, request.score,
                )
            except Exception as ls_err:
                logger.warning("LangSmith feedback failed (saving locally): %s", ls_err)

        # ── 2. Local redundancy log ──────────────────────────────────
        entry = {
            "run_id": request.run_id,
            "score": request.score,
            "comment": request.comment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(FEEDBACK_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as file_err:
            logger.warning("Local feedback log failed: %s", file_err)

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")




if __name__ == "__main__":
    import uvicorn

    PORT = int(os.getenv("PORT", "8000"))
    _free_port(PORT)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        access_log=False,           # We handle access logging ourselves
        timeout_keep_alive=30,
    )
