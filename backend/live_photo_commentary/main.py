import asyncio
import json
import logging
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from . import config
from .pipeline import Pipeline

log = logging.getLogger(__name__)

# Single active WebSocket connection (one Tauri webview, one server).
_ws: WebSocket | None = None

pipeline: Pipeline | None = None
_model_errors: dict[str, str] = {}      # "vlm" | "tts" → last error message
_model_local_dirs: dict[str, str] = {}  # model_id → local download path (no symlinks)
_vlm_ready: bool = False
_tts_ready: bool = False


async def _send(data: dict) -> None:
    if _ws is not None:
        try:
            await _ws.send_text(json.dumps(data))
        except Exception:
            pass


def _send_threadsafe(data: dict, loop: asyncio.AbstractEventLoop) -> None:
    asyncio.run_coroutine_threadsafe(_send(data), loop)


async def _send_ready_state() -> None:
    await _send({
        "type": "ready_state",
        "ready": _vlm_ready and _tts_ready,
        "vlm": _vlm_ready,
        "tts": _tts_ready,
    })


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
    {"provider": "local", "model_id": "google/gemma-4-E4B-it"},
    {"provider": "local", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct"},
    {"provider": "local", "model_id": "Qwen/Qwen2.5-VL-7B-Instruct"},
    {"provider": "local", "model_id": "apple/FastVLM-0.5B"},
    {"provider": "local", "model_id": "apple/FastVLM-1.5B"},
    {"provider": "local", "model_id": "apple/FastVLM-7B"},
]


def _make_describer(on_progress=None):
    cfg = config.get()
    if cfg.vlm_provider == "local":
        model_id = cfg.vlm_model or "microsoft/Phi-4-multimodal-instruct"
        path = _model_local_dirs.get(model_id, model_id)
        from .local_describer import LocalDescriber
        return LocalDescriber(model_id=path, on_progress=on_progress)
    from .remote_describer import RemoteDescriber
    return RemoteDescriber(provider=cfg.vlm_provider, model_id=cfg.vlm_model)


def _make_synthesizer():
    import json
    from .synthesizers.kokoro import KokoroSynthesizer
    raw = config.get().tts_voice
    try:
        voice = json.loads(raw)  # e.g. {"af_nicole": 0.8, "jf_alpha": 0.2}
    except (json.JSONDecodeError, TypeError):
        voice = raw              # plain voice name
    return KokoroSynthesizer(voice=voice)


async def _download_with_progress(model_id: str) -> str:
    """Download model files to a flat local_dir (no symlinks) with progress events.
    Returns the local directory path for use with from_pretrained."""
    import os
    import tqdm as tqdm_lib
    from huggingface_hub import snapshot_download, constants

    local_dir = str(Path(constants.HF_HUB_CACHE) / "lpc" / model_id.replace("/", "--"))
    loop = asyncio.get_running_loop()

    class _ProgressTqdm(tqdm_lib.tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._last_n = 0

        def update(self, n=1):
            super().update(n)
            if not self.total:
                return
            if self.n - self._last_n >= self.total * 0.01 or self.n >= self.total:
                self._last_n = self.n
                desc = self.desc or ""
                filename = os.path.basename(desc)
                if not filename or " " in filename:
                    filename = model_id.split("/")[-1]
                _send_threadsafe({
                    "type": "download_progress",
                    "file": filename,
                    "downloaded": self.n,
                    "total": self.total,
                }, loop)

    await asyncio.to_thread(
        snapshot_download,
        repo_id=model_id,
        local_dir=local_dir,
        tqdm_class=_ProgressTqdm,
        ignore_patterns=["*.msgpack", "*.h5", "flax_*", "tf_*", "rust_model*"],
    )
    return local_dir


async def _reinit_describer() -> None:
    global _vlm_ready
    assert pipeline is not None
    pipeline.describer = None
    _model_errors.pop("vlm", None)
    _vlm_ready = False
    await _send({"type": "model_loading", "model": "vlm"})
    await _send_ready_state()
    try:
        cfg = config.get()
        loop = asyncio.get_running_loop()

        def _progress(msg):
            _send_threadsafe(msg, loop)

        if cfg.vlm_provider == "local":
            model_id = cfg.vlm_model or "microsoft/Phi-4-multimodal-instruct"
            _model_local_dirs[model_id] = await _download_with_progress(model_id)
        pipeline.describer = await asyncio.to_thread(_make_describer, _progress)
        log.info("Describer ready")
        _vlm_ready = True
        await _send({"type": "model_ready", "model": "vlm"})
        await _send_ready_state()
    except Exception as exc:
        msg = f"VLM init failed: {exc}"
        _model_errors["vlm"] = msg
        log.warning(msg)
        await _send({"type": "error", "message": msg})
        await _send_ready_state()


async def _reinit_synthesizer() -> None:
    global _tts_ready
    assert pipeline is not None
    pipeline.synthesizer = None
    _model_errors.pop("tts", None)
    _tts_ready = False
    await _send({"type": "model_loading", "model": "tts"})
    await _send_ready_state()
    try:
        pipeline.synthesizer = await asyncio.to_thread(_make_synthesizer)
        log.info("Synthesizer ready")
        _tts_ready = True
        await _send({"type": "model_ready", "model": "tts"})
        await _send_ready_state()
    except Exception as exc:
        msg = f"TTS init failed: {exc}"
        _model_errors["tts"] = msg
        log.warning(msg)
        await _send({"type": "error", "message": msg})
        await _send_ready_state()


async def _init_models() -> None:
    await asyncio.gather(_reinit_describer(), _reinit_synthesizer())


def _suppress_pipe_reset(loop, context):
    # WinError 10054 on pipe shutdown is benign noise from taskkill teardown.
    if isinstance(context.get("exception"), ConnectionResetError):
        return
    loop.default_exception_handler(context)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    loop = asyncio.get_running_loop()
    if sys.platform == "win32":
        loop.set_exception_handler(_suppress_pipe_reset)
    pipeline = Pipeline()
    pipeline.attach(loop, _send)
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


async def _send_models() -> None:
    from .synthesizers.kokoro import KokoroSynthesizer
    try:
        voices = await asyncio.to_thread(KokoroSynthesizer.list_voices)
    except Exception as exc:
        log.warning("Could not list Kokoro voices: %s", exc)
        voices = []
    await _send({
        "type": "models",
        "vlm": _VLM_CATALOGUE,
        "tts": [{"engine": "kokoro", "voice": v} for v in voices],
    })


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global _ws
    await ws.accept()
    _ws = ws
    await _send({"type": "config", "data": config.as_dict()})
    await _send_ready_state()
    for msg in _model_errors.values():
        await _send({"type": "error", "message": msg})
    asyncio.create_task(_send_models())
    try:
        while True:
            data = await ws.receive_json()
            match data.get("type"):
                case "get_config":
                    await _send({"type": "config", "data": config.as_dict()})
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
                    await _send({"type": "config", "data": config.as_dict()})
                case "get_models":
                    asyncio.create_task(_send_models())
                case "start_cycle":
                    if pipeline is None:
                        await _send({"type": "error", "message": "Pipeline not ready"})
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
        if _ws is ws:
            _ws = None
            # Stop the pipeline so start_cycle works cleanly on reconnect.
            if pipeline is not None:
                await pipeline.stop_loop()
