"""FastAPI application exposing Synn Core controls."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore
    BaseModel = object  # type: ignore

LOGGER = logging.getLogger(__name__)


def create_app(core: "SynnCore") -> "FastAPI":
    """Instantiate FastAPI app bound to the running core."""

    if not FastAPI:
        raise RuntimeError("FastAPI is not installed")

    app = FastAPI(title="Synn Core API", description="Control interface for the Synn assistant")
    main_loop = core.event_loop

    class SpeakRequest(BaseModel):
        text: str
        tone: Optional[float] = None
        pitch: Optional[float] = None
        speed: Optional[float] = None

    class RememberRequest(BaseModel):
        topic: str
        content: str

    @app.post("/speak")
    async def speak(request: SpeakRequest) -> Dict[str, Any]:
        if not main_loop:
            raise HTTPException(status_code=503, detail="Core loop not available")
        coro = core.speech.speak(request.text, tone=request.tone or 0.0, pitch=request.pitch or 0.0, speed=request.speed or 1.0)
        asyncio.run_coroutine_threadsafe(coro, main_loop)
        return {"status": "queued"}

    @app.get("/emotion")
    async def emotion() -> Dict[str, Any]:
        state = core.emotion_engine.current_state
        return {"emotion": state.name, "valence": state.valence, "arousal": state.arousal}

    @app.post("/remember")
    async def remember(request: RememberRequest) -> Dict[str, Any]:
        if not main_loop:
            raise HTTPException(status_code=503, detail="Core loop not available")
        coro = core.long_term_memory.add_knowledge(request.topic, request.content)
        asyncio.run_coroutine_threadsafe(coro, main_loop)
        return {"status": "queued"}

    @app.get("/analyze_face")
    async def analyze_face() -> Dict[str, Any]:
        if not main_loop:
            raise HTTPException(status_code=503, detail="Core loop not available")
        future = asyncio.run_coroutine_threadsafe(core.camera_sensor.capture(), main_loop)
        try:
            result = future.result(timeout=5)
        except Exception as exc:  # pragma: no cover - runtime timing issues
            raise HTTPException(status_code=500, detail=str(exc))
        return {"emotion": result}

    return app


__all__ = ["create_app"]
