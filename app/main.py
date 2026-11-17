"""
Application entry point — Stage 1.

Creates the FastAPI app, registers the API routers, and exposes a health check.
Run it with:

    .venv/bin/uvicorn app.main:app --reload --port 8000

Then open http://localhost:8000/docs for the auto-generated interactive API docs.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import config
from app.api import upload, chat, search, agent

app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="Ask questions about your own documents, answered by a local LLM. No cloud.",
)

# Register the routers defined in app/api/*.
app.include_router(upload.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(agent.router)


@app.get("/health", tags=["system"])
async def health():
    """Liveness probe + a glance at how the app is configured."""
    return {
        "status": "ok",
        "app": config.APP_NAME,
        "version": config.APP_VERSION,
        "llm_base_url": config.LLM_BASE_URL,
        "upload_dir": config.rel_to_base(config.UPLOAD_DIR),
    }


# Serve the web chat UI at /ui (static HTML/JS, same origin as the API).
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if _WEB_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_WEB_DIR), html=True), name="ui")


@app.get("/", tags=["system"])
async def root():
    return {"message": f"{config.APP_NAME} v{config.APP_VERSION}", "ui": "/ui", "docs": "/docs"}
