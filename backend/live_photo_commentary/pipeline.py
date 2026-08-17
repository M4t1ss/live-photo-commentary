import asyncio
import queue
import threading
from pathlib import Path
from tempfile import mkdtemp

from PIL import Image

from . import config

_SENTINEL = object()


class Pipeline:
    def __init__(self):
        self.describer = None
        self.synthesizer = None

        self._tmp_dir = Path(mkdtemp(prefix="lpc_"))
        self._frame_path: Path | None = None
        self._prev_img = None
        self._last_rejected = False
        self._audio_chunks: dict[int, Path] = {}

        self._loop: asyncio.AbstractEventLoop | None = None
        self._broadcast_fn = None

        self._auto_loop_running = False
        self._auto_loop_task: asyncio.Task | None = None
        self._frame_q: asyncio.Queue | None = None  # created on start_loop()

        self._generation: int = 0
        self._vlm_q: queue.Queue = queue.Queue(maxsize=1)
        self._tts_q: queue.Queue = queue.Queue(maxsize=1)
        self._describer_lock = threading.Lock()

        threading.Thread(target=self._vlm_worker, daemon=True).start()
        threading.Thread(target=self._tts_worker, daemon=True).start()

    def take_describer_for_reinit(self):
        """Atomically detach the current describer so it's safe to free.

        Uses the same lock the VLM worker holds while calling the describer, so
        this blocks until any in-flight generation has actually finished before
        handing the (now-detached) describer back to the caller for cleanup.
        """
        with self._describer_lock:
            old = self.describer
            self.describer = None
            return old

    def attach(self, broadcast_fn) -> None:
        self._broadcast_fn = broadcast_fn

    async def trigger(self, path: str) -> None:
        prev_img = self._prev_img

        def _load():
            img = Image.open(path)
            img.load()
            return img

        img = await asyncio.to_thread(_load)

        self._prev_img = img
        self._frame_path = Path(path)

        diff = None
        accepted = True
        if prev_img is not None:
            cfg = config.get()
            if cfg.difference_threshold > 0:
                from .screenshot import difference
                diff = await asyncio.to_thread(
                    difference, img, prev_img, cfg.difference_measure
                )
                accepted = diff >= cfg.difference_threshold

        # Only push the displayed frame into "previous" when the currently
        # shown frame was itself accepted; consecutive rejections (and the
        # accepted frame that ends a rejection streak) replace it in place.
        push = not self._last_rejected
        self._last_rejected = not accepted

        self._generation += 1
        if self._broadcast_fn:
            frame_msg = {"type": "frame", "url": "/frame/current", "push": push, "gen": self._generation}
            if diff is not None:
                frame_msg["diff"] = round(diff, 6)
                frame_msg["measure"] = cfg.difference_measure
            await self._broadcast_fn(frame_msg)

        if not accepted:
            if self._broadcast_fn:
                await self._broadcast_fn({"type": "skipped", "diff": round(diff, 6)})
            # Ask JS to take another screenshot so the loop can retry.
            self._send({"type": "take_screenshot"})
            return

        try:
            self._vlm_q.put_nowait((self._generation, img, prev_img))
        except queue.Full:
            if self._broadcast_fn:
                await self._broadcast_fn({"type": "busy", "message": "VLM busy, cycle skipped"})

    @property
    def frame_path(self) -> Path | None:
        return self._frame_path

    @property
    def running(self) -> bool:
        return self._auto_loop_running

    def chunk_path(self, index: int) -> Path | None:
        return self._audio_chunks.get(index)

    async def start_loop(self) -> None:
        if self._auto_loop_running:
            return
        self._auto_loop_running = True
        self._loop = asyncio.get_running_loop()
        self._frame_q = asyncio.Queue()
        self._auto_loop_task = asyncio.create_task(self._auto_loop())

    async def stop_loop(self) -> None:
        self._auto_loop_running = False
        self._generation += 1
        if self._auto_loop_task:
            self._auto_loop_task.cancel()
            self._auto_loop_task = None

    def on_frame_ready(self, path: str) -> None:
        if self._loop and self._frame_q:
            self._loop.call_soon_threadsafe(self._frame_q.put_nowait, path)

    async def _auto_loop(self) -> None:
        while self._auto_loop_running:
            path = await self._frame_q.get()
            if not self._auto_loop_running:
                break
            try:
                await self.trigger(path)
            except Exception as exc:
                self._send({"type": "error", "message": f"Cycle error: {exc}"})

    def _send(self, data: dict) -> None:
        if self._loop and self._broadcast_fn:
            asyncio.run_coroutine_threadsafe(self._broadcast_fn(data), self._loop)

    def _vlm_worker(self) -> None:
        while True:
            try:
                item = self._vlm_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                return
            gen, curr_img, prev_img = item
            if gen != self._generation:
                continue
            with self._describer_lock:
                describer = self.describer
                if describer is None:
                    self._send({"type": "error", "message": "No VLM configured"})
                    continue
                describer.max_history_size = config.get().max_history_size
                try:
                    text = describer(curr_img, prev_img)
                except Exception as exc:
                    self._send({"type": "error", "message": f"VLM error: {exc}"})
                    continue
                finally:
                    describer = None
            if gen != self._generation:
                continue
            try:
                self._tts_q.put_nowait((gen, text))
            except queue.Full:
                self._send({"type": "error", "message": "TTS busy"})

    def _tts_worker(self) -> None:
        while True:
            try:
                item = self._tts_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                return
            gen, text = item
            if gen != self._generation:
                continue
            if self.synthesizer is None:
                self._send({"type": "error", "message": "No TTS configured"})
                continue
            try:
                self._audio_chunks.clear()
                for i, (audio, fragment, chunk_phonemes, mark_timings, subtitle_segments) in enumerate(self.synthesizer(text)):
                    if gen != self._generation:
                        break
                    audio_path = self._tmp_dir / f"chunk_{i}.wav"
                    audio_path.write_bytes(self.synthesizer.to_wav_bytes(audio))
                    self._audio_chunks[i] = audio_path
                    sub_times = [round(t, 4) for name, t in mark_timings if name == "sub"]
                    subtitles = [
                        {"text": seg, "time": t}
                        for seg, t in zip(subtitle_segments, [0.0] + sub_times)
                    ]
                    tags = [
                        {"name": name, "time": round(t, 4)}
                        for name, t in mark_timings if name != "sub"
                    ]
                    self._send({
                        "type": "chunk",
                        "gen": gen,
                        "index": i,
                        "text": fragment,
                        "audio_url": f"/audio/chunk/{i}",
                        "phonemes": [[ph, round(t, 4)] for ph, t in chunk_phonemes],
                        "subtitles": subtitles,
                        "tags": tags,
                    })
                else:
                    self._send({"type": "tts_done", "gen": gen})
            except Exception as exc:
                self._send({"type": "error", "message": f"TTS error: {exc}"})

    def stop(self) -> None:
        self._auto_loop_running = False
        if self._auto_loop_task:
            self._auto_loop_task.cancel()
            self._auto_loop_task = None
        for q in (self._vlm_q, self._tts_q):
            try:
                q.put_nowait(_SENTINEL)
            except queue.Full:
                pass  # daemon threads will exit with the process

    def cleanup(self) -> None:
        import shutil
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
