# ABOUTME: Lightweight FastAPI server that serves the blob UI and provides a chat API
# ABOUTME: Bridges the standalone 3D blob interface with the Heathcliff agent

import os
import sys
import uuid
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import Config
from core.agent_core import HeathcliffAgent
from core.runtime.http_client import RuntimeV2HttpClient
from db.memory_manager import MemoryManager
from logger import logger

# ── Blob config ───────────────────────────────────────────────────────
ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
)
BLOB_DEFAULT_PORT = 8600
BLOB_STATES = ("idle", "listening", "thinking", "speaking")

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Heathcliff Blob UI", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Agent singleton ───────────────────────────────────────────────────
_memory: MemoryManager | None = None
_agent: Any | None = None
_conversation_id: str = str(uuid.uuid4())


def _get_agent() -> Any:
    """Lazy-init the Heathcliff agent."""
    global _memory, _agent
    if _agent is None:
        logger.info("Initializing Heathcliff agent for blob UI...")
        if Config.RUNTIME_V2_ENABLED:
            _agent = RuntimeV2HttpClient(Config.RUNTIME_V2_URL)
        else:
            _memory = MemoryManager()
            _agent = HeathcliffAgent(memory_manager=_memory)
        logger.info("Agent ready.")
    return _agent


# ── Request / Response models ─────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class StateRequest(BaseModel):
    state: str


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    """Serve the blob UI."""
    logger.info("Serving blob UI")
    if _agent is None:
        logger.info("Initializing Heathcliff...")
    _get_agent()
    logger.info("Blob UI ready.")
    return FileResponse(os.path.join(ASSETS_DIR, "index.html"))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a message to Heathcliff and get a response."""
    if _agent is None:
        logger.info("Initializing Heathcliff...")
    agent = _get_agent()
    logger.info("Blob UI ready.")
    try:
        logger.info(f"Blob UI request: {req.message}")
        response = agent.invoke(req.message, conversation_id=_conversation_id)
        logger.info(f"Blob UI response: {response}")
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Blob UI error: {e}")
        return ChatResponse(response=f"Sorry, I encountered an error: {str(e)}")


@app.post("/api/state")
async def set_state(req: StateRequest):
    """Set the blob animation state (for external callers like voice pipeline)."""
    if req.state not in BLOB_STATES:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Invalid state '{req.state}'. Must be one of: {list(BLOB_STATES)}"
            },
        )
    # State is applied client-side; this endpoint is a validated relay.
    # Clients poll or use SSE in a future iteration.
    logger.info(f"Blob state set to: {req.state}")
    return {"state": req.state}


@app.get("/api/states")
async def list_states():
    """Return the list of valid blob states."""
    return {"states": list(BLOB_STATES)}


# Mount static files AFTER explicit routes so they don't shadow /api/*
app.mount("/", StaticFiles(directory=ASSETS_DIR, html=True), name="static")


# ── Entrypoint ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("BLOB_PORT", BLOB_DEFAULT_PORT))
    logger.info(f"Starting Heathcliff Blob UI on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
