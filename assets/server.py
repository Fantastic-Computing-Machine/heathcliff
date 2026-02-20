# ABOUTME: Lightweight FastAPI server that serves the blob UI and provides a chat API
# ABOUTME: Bridges the standalone 3D blob interface with the Heathcliff agent

import os
import sys
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import Config
from core.agent_core import HeathcliffAgent
from core.memory_manager import MemoryManager
from logger import logger

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Heathcliff Blob UI", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static assets (serves index.html, css, js from this directory) ───
ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Agent singleton ───────────────────────────────────────────────────
_memory: MemoryManager | None = None
_agent: HeathcliffAgent | None = None
_session_id: str = str(uuid.uuid4())


def _get_agent() -> HeathcliffAgent:
    """Lazy-init the Heathcliff agent."""
    global _memory, _agent
    if _agent is None:
        logger.info("Initializing Heathcliff agent for blob UI...")
        _memory = MemoryManager()
        _agent = HeathcliffAgent(memory_manager=_memory)
        logger.info("Agent ready.")
    return _agent


# ── Request / Response models ─────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    """Serve the blob UI."""
    return FileResponse(os.path.join(ASSETS_DIR, "index.html"))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a message to Heathcliff and get a response."""
    agent = _get_agent()
    try:
        logger.info(f"Blob UI request: {req.message}")
        response = agent.invoke(req.message, session_id=_session_id)
        logger.info(f"Blob UI response: {response}")
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Blob UI error: {e}")
        return ChatResponse(response=f"Sorry, I encountered an error: {str(e)}")


# Mount static files AFTER explicit routes so they don't shadow /api/*
app.mount("/", StaticFiles(directory=ASSETS_DIR, html=True), name="static")


# ── Entrypoint ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("BLOB_PORT", 8600))
    logger.info(f"Starting Heathcliff Blob UI on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
