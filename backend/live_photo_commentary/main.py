import asyncio
import logging
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from . import config
from .connections import ConnectionManager
from .pipeline import Pipeline

log = logging.getLogger(__name__)

manager = ConnectionManager()
pipeline: Pipeline | None = None

_VLM_CATALOGUE = [
    # Cloud
    {"provider": "gemini", "model_id": "gemini-2.0-flash-exp"},
    {"provider": "gemini", "model_id": "gemini-2.5-flash-lite-preview-06-17"},
    {"provider": "openai",  "model_id": "gpt-4o"},
    {"provider": "openai",  "model_id": "gpt-4o-mini"},
    # Local
    {"provider": "local", "model_id": "microsoft/Phi-4-multimodal-instruct"},
    {"provider": "local", "model_id": "google/gemma-3-4b-it"},
    {"provider": "local", "model_id": "google/gemma-3-12b-it"},
    {"provider": "local", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct"},
    {"provider": "local", "model_id": "Qwen/Qwen2.5-VL-7B-Instruct"},
    {"provider": "local", "model_id": "apple/FastVLM-0.5B"},
    {"provider": "local", "model_id": "apple/FastVLM-1.5B"},
    {"provider": "local", "model_id": "apple/FastVLM-7B"},
]


def _make_describer():
    cfg = config.get()
    if cfg.vlm_provider == "local":
        from .local_describer import LocalDescriber
        return LocalDescriber(model_id=cfg.vlm_model or "microsoft/Phi-4-multimodal-instruct")
    from .remote_describer import RemoteDescriber
    return RemoteDescriber(provider=cfg.vlm_provider, model_id=cfg.vlm_model)


def _make_synthesizer():
    from .synthesizers.kokoro import KokoroSynthesizer
    return KokoroSynthesizer(voice=config.get().tts_voice)


async def _reinit_describer() -> None:
    assert pipeline is not None
    pipeline.describer = None
    try:
        pipeline.describer = await asyncio.to_thread(_make_describer)
        log.info("Describer ready")
        await manager.broadcast({"type": "model_ready", "model": "vlm"})
    except Exception as exc:
        log.warning("Describer init failed: %s", exc)
        await manager.broadcast({"type": "error", "message": f"VLM init failed: {exc}"})


async def _reinit_synthesizer() -> None:
    assert pipeline is not None
    pipeline.synthesizer = None
    try:
        pipeline.synthesizer = await asyncio.to_thread(_make_synthesizer)
        log.info("Synthesizer ready")
        await manager.broadcast({"type": "model_ready", "model": "tts"})
    except Exception as exc:
        log.warning("Synthesizer init failed: %s", exc)
        await manager.broadcast({"type": "error", "message": f"TTS init failed: {exc}"})


async def _init_models() -> None:
    await asyncio.gather(_reinit_describer(), _reinit_synthesizer())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = Pipeline()
    pipeline.attach(asyncio.get_running_loop(), manager.broadcast)
    asyncio.create_task(_init_models())
    yield
    pipeline.stop()
    pipeline.cleanup()


app = FastAPI(lifespan=lifespan)



@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/frame/current")
async def get_current_frame():
    path = pipeline.frame_path if pipeline else None
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="No frame available")
    return FileResponse(str(path), media_type="image/png")


@app.get("/audio/chunk/{index}")
async def get_audio_chunk(index: int):
    path = pipeline.chunk_path(index) if pipeline else None
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Chunk not available")
    return FileResponse(str(path), media_type="audio/wav")


async def _broadcast_models() -> None:
    from .synthesizers.kokoro import KokoroSynthesizer
    try:
        voices = await asyncio.to_thread(KokoroSynthesizer.list_voices)
    except Exception as exc:
        log.warning("Could not list Kokoro voices: %s", exc)
        voices = []
    await manager.broadcast({
        "type": "models",
        "vlm": _VLM_CATALOGUE,
        "tts": [{"engine": "kokoro", "voice": v} for v in voices],
    })


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_json()
            match data.get("type"):
                case "get_config":
                    await manager.broadcast({"type": "config", "data": config.as_dict()})
                case "set_config":
                    old = config.get()
                    config.apply(data.get("data", {}))
                    new = config.get()
                    if data.get("persist"):
                        await asyncio.to_thread(config.persist, Path(".env"))
                    if pipeline is not None:
                        if old.vlm_provider != new.vlm_provider or old.vlm_model != new.vlm_model:
                            asyncio.create_task(_reinit_describer())
                        if old.tts_voice != new.tts_voice:
                            asyncio.create_task(_reinit_synthesizer())
                    await manager.broadcast({"type": "config", "data": config.as_dict()})
                case "get_models":
                    asyncio.create_task(_broadcast_models())
                case "start_cycle":
                    if pipeline is None:
                        await ws.send_json({"type": "error", "message": "Pipeline not ready"})
                    else:
                        await pipeline.start_loop()
                case "stop_cycle":
                    if pipeline is not None:
                        await pipeline.stop_loop()
                case "speech_ended":
                    if pipeline is not None:
                        pipeline.on_speech_ended()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(ws)
