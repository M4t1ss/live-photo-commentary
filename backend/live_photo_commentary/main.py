import asyncio
import json
import yaml
import logging
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Piped stdout/stderr fall back to the legacy console codepage (e.g. cp932),
    # which can't encode arbitrary VLM output (non-Latin scripts, emoji, etc.).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import config, prompts
from .pipeline import Pipeline
from .prompts import DEFAULT_NAME as _DEFAULT_PROMPTSET

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
        text = json.dumps(data)
        try:
            await _ws.send_text(text)
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
        "running": pipeline.running if pipeline is not None else False,
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
    {"provider": "local", "model_id": "google/gemma-4-E2B-it"},
    {"provider": "local", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct"},
    {"provider": "local", "model_id": "Qwen/Qwen2.5-VL-7B-Instruct"},
    {"provider": "local", "model_id": "apple/FastVLM-0.5B"},
    {"provider": "local", "model_id": "apple/FastVLM-1.5B"},
    {"provider": "local", "model_id": "apple/FastVLM-7B"},
]


def _free_describer(describer) -> None:
    """Drop a describer's model/processor and force the GPU memory to be released
    before the next model loads, since accelerate's automatic device placement can
    otherwise offload part of the new model to cpu/disk if it still sees the old
    model's memory as in use."""
    if describer is None:
        return
    for attr in ("model", "processor", "tokenizer"):
        if hasattr(describer, attr):
            setattr(describer, attr, None)
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _make_describer(on_progress=None):
    cfg = config.get()
    if cfg.vlm_provider == "local":
        path = _model_local_dirs.get(cfg.vlm_model, cfg.vlm_model)
        from .local_describer import LocalDescriber
        describer = LocalDescriber(model_id=path, on_progress=on_progress)
    else:
        from .remote_describer import RemoteDescriber
        api_key = {"gemini": cfg.gemini_api_key, "openai": cfg.openai_api_key}.get(cfg.vlm_provider)
        describer = RemoteDescriber(provider=cfg.vlm_provider, model_id=cfg.vlm_model, api_key=api_key)
    return describer


def _load_tag_names() -> list[str]:
    yaml_path = config.get().model_dir / "model.yaml"
    if not yaml_path.exists():
        return []
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return list(raw.get("tags", {}).keys())


def _apply_promptset(name: str) -> None:
    """Update the live describer's prompt fields without reloading the model."""
    if pipeline is not None and pipeline.describer is not None:
        fields = prompts.substitute_tags(prompts.load(name), _load_tag_names())
        for k, v in fields.items():
            setattr(pipeline.describer, k, v)


def _make_synthesizer():
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
    _model_errors.pop("vlm", None)
    _vlm_ready = False
    cfg = config.get()
    if not cfg.vlm_provider or not cfg.vlm_model:
        # Fresh install / no model chosen yet — don't download or connect to
        # anything until the user picks one in Settings and saves.
        await _send_ready_state()
        return
    await _send({"type": "model_loading", "model": "vlm"})
    await _send_ready_state()
    try:
        # Detach the current describer. This blocks (off the event loop thread)
        # until any in-flight generation using it has actually finished, then
        # frees it, so the new model never loads while the old one is still
        # holding GPU memory (see take_describer_for_reinit / _free_describer).
        if pipeline.describer is not None:
            await _send({"type": "load_progress", "message": "Waiting for current generation to finish…"})
        old_describer = await asyncio.to_thread(pipeline.take_describer_for_reinit)
        if old_describer is not None:
            await _send({"type": "load_progress", "message": "Releasing previous model…"})
            await asyncio.to_thread(_free_describer, old_describer)

        loop = asyncio.get_running_loop()

        def _progress(msg):
            _send_threadsafe(msg, loop)

        if cfg.vlm_provider == "local":
            _model_local_dirs[cfg.vlm_model] = await _download_with_progress(cfg.vlm_model)
        pipeline.describer = await asyncio.to_thread(_make_describer, _progress)
        _apply_promptset(config.get().active_promptset)
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
    prompts.set_dir(Path("prompts"))
    loop = asyncio.get_running_loop()
    if sys.platform == "win32":
        loop.set_exception_handler(_suppress_pipe_reset)
    pipeline = Pipeline()
    pipeline.attach(_send)
    asyncio.create_task(_init_models())
    yield
    pipeline.stop()
    pipeline.cleanup()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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


@app.get("/model-config")
async def get_model_config():
    yaml_path = config.get().model_dir / "model.yaml"
    if not yaml_path.exists():
        return {}
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {
        "jawBone": raw.get("jaw_bone"),
        "jawAxis": raw.get("jaw_axis", "x"),
        "minJawAngle": raw.get("min_jaw_angle", 0.0),
        "maxJawAngle": raw.get("max_jaw_angle", 0.15),
        "neckBone": raw.get("neck_bone"),
        "maxNeckAngle": raw.get("max_neck_angle", 0.3),
        "headBone": raw.get("head_bone"),
        "maxHeadAngle": raw.get("max_head_angle", 0.5),
        "leftEyeBone": raw.get("left_eye_bone"),
        "rightEyeBone": raw.get("right_eye_bone"),
        "maxEyeAngle": raw.get("max_eye_angle", 0.4),
        "maxEnvelopeDuration": raw.get("max_envelope_duration", 0.3),
        "visemeMap": raw.get("viseme_map", {}),
        "blink": raw.get("blink"),
        "tags": raw.get("tags", {}),
        "idleAnimation": raw.get("animations", {}).get("idle"),
        "talkingAnimation": raw.get("animations", {}).get("talking"),
        "lighting": raw.get("lighting"),
    }


@app.get("/model")
async def get_model():
    model_path = config.get().model_dir / "model.glb"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(str(model_path), media_type="model/gltf-binary")


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
    active = config.get().active_promptset
    await _send({
        "type": "promptset_loaded",
        "name": active,
        "fields": prompts.load(active),
        "names": prompts.list_names(),
    })
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
                        api_key_changed = (
                            (new.vlm_provider == "gemini" and old.gemini_api_key != new.gemini_api_key)
                            or (new.vlm_provider == "openai" and old.openai_api_key != new.openai_api_key)
                        )
                        if old.vlm_provider != new.vlm_provider or old.vlm_model != new.vlm_model or api_key_changed:
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
                case "frame_ready":
                    if pipeline is not None:
                        pipeline.on_frame_ready(data.get("path", ""))
                case "list_promptsets":
                    await _send({"type": "promptsets", "names": prompts.list_names()})
                case "load_promptset":
                    name = (data.get("name") or "").strip() or _DEFAULT_PROMPTSET
                    fields = prompts.load(name)
                    config.apply({"active_promptset": name})
                    await asyncio.to_thread(config.persist, Path(".env"))
                    _apply_promptset(name)
                    if pipeline is not None and pipeline.describer is not None:
                        pipeline.describer.reset()
                    await _send({
                        "type": "promptset_loaded",
                        "name": name,
                        "fields": fields,
                        "names": prompts.list_names(),
                    })
                case "save_promptset":
                    name = (data.get("name") or "").strip()
                    if not name or name == _DEFAULT_PROMPTSET:
                        await _send({"type": "error", "message": "Cannot save as 'default'"})
                    else:
                        prompts.save(name, data.get("fields", {}))
                        if name == config.get().active_promptset:
                            _apply_promptset(name)
                        await _send({"type": "promptsets", "names": prompts.list_names()})
                case "delete_promptset":
                    name = (data.get("name") or "").strip()
                    if not name or name == _DEFAULT_PROMPTSET:
                        await _send({"type": "error", "message": "Cannot delete 'default'"})
                    else:
                        was_active = config.get().active_promptset == name
                        prompts.delete(name)
                        if was_active:
                            config.apply({"active_promptset": _DEFAULT_PROMPTSET})
                            await asyncio.to_thread(config.persist, Path(".env"))
                            _apply_promptset(_DEFAULT_PROMPTSET)
                        active = config.get().active_promptset
                        await _send({
                            "type": "promptset_loaded",
                            "name": active,
                            "fields": prompts.load(active),
                            "names": prompts.list_names(),
                        })
    except WebSocketDisconnect:
        pass
    finally:
        if _ws is ws:
            _ws = None
            # Stop the pipeline so start_cycle works cleanly on reconnect.
            if pipeline is not None:
                await pipeline.stop_loop()
