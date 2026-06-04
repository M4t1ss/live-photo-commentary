import asyncio
import queue
import threading
from pathlib import Path
from tempfile import mkdtemp

from . import config

_SENTINEL = object()


class Pipeline:
    def __init__(self):
        self.describer = None
        self.synthesizer = None

        self._tmp_dir = Path(mkdtemp(prefix="lpc_"))
        self._frame_slot = -1
        self._frame_path: Path | None = None
        self._prev_img = None
        self._audio_chunks: dict[int, Path] = {}

        self._loop: asyncio.AbstractEventLoop | None = None
        self._broadcast_fn = None

        self._auto_loop_running = False
        self._auto_loop_task: asyncio.Task | None = None
        self._speech_ended: asyncio.Event | None = None  # created on attach()

        self._vlm_q: queue.Queue = queue.Queue(maxsize=1)
        self._tts_q: queue.Queue = queue.Queue(maxsize=1)

        threading.Thread(target=self._vlm_worker, daemon=True).start()
        threading.Thread(target=self._tts_worker, daemon=True).start()

    def attach(self, loop: asyncio.AbstractEventLoop, broadcast_fn) -> None:
        self._loop = loop
        self._broadcast_fn = broadcast_fn
        self._speech_ended = asyncio.Event()

    async def trigger(self) -> None:
        from .screenshot import screenshot

        prev_img = self._prev_img

        # Alternate between two file slots so the HTTP endpoint never serves a
        # file that is being overwritten.
        new_slot = 0 if self._frame_slot != 0 else 1
        new_path = self._tmp_dir / f"frame_{new_slot}.png"
        img = await asyncio.to_thread(screenshot, new_path)

        self._prev_img = img
        self._frame_path = new_path
        self._frame_slot = new_slot

        await self._broadcast_fn({"type": "frame", "url": "/frame/current"})

        if prev_img is not None:
            cfg = config.get()
            if cfg.difference_threshold > 0:
                from .screenshot import difference
                diff = await asyncio.to_thread(
                    difference, img, prev_img, cfg.difference_measure
                )
                if diff < cfg.difference_threshold:
                    await self._broadcast_fn({"type": "skipped", "diff": round(diff, 6)})
                    self.on_speech_ended()  # unblock the loop so it retries
                    return

        try:
            self._vlm_q.put_nowait((img, prev_img))
        except queue.Full:
            await self._broadcast_fn({"type": "busy", "message": "VLM busy, cycle skipped"})

    @property
    def frame_path(self) -> Path | None:
        return self._frame_path

    def chunk_path(self, index: int) -> Path | None:
        return self._audio_chunks.get(index)

    async def start_loop(self) -> None:
        if self._auto_loop_running:
            return
        self._auto_loop_running = True
        self._speech_ended.clear()
        await self.trigger()
        self._auto_loop_task = asyncio.create_task(self._auto_loop())

    async def stop_loop(self) -> None:
        self._auto_loop_running = False
        if self._auto_loop_task:
            self._auto_loop_task.cancel()
            self._auto_loop_task = None

    def on_speech_ended(self) -> None:
        if self._speech_ended:
            self._speech_ended.set()

    async def _auto_loop(self) -> None:
        while self._auto_loop_running:
            await self._speech_ended.wait()
            self._speech_ended.clear()
            if not self._auto_loop_running:
                break
            await asyncio.sleep(config.get().post_speech_delay)
            if not self._auto_loop_running:
                break
            try:
                await self.trigger()
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
            describer = self.describer
            if describer is None:
                self._send({"type": "error", "message": "No VLM configured"})
                continue
            curr_img, prev_img = item
            describer.max_history_size = config.get().max_history_size
            try:
                text = describer(curr_img, prev_img)
            except Exception as exc:
                self._send({"type": "error", "message": f"VLM error: {exc}"})
                continue
            try:
                self._tts_q.put_nowait(text)
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
            if self.synthesizer is None:
                self._send({"type": "error", "message": "No TTS configured"})
                continue
            text = item
            try:
                self._audio_chunks.clear()
                for i, (audio, fragment, chunk_phonemes, _mark_timings) in enumerate(self.synthesizer(text)):
                    audio_path = self._tmp_dir / f"chunk_{i}.wav"
                    audio_path.write_bytes(self.synthesizer.to_wav_bytes(audio))
                    self._audio_chunks[i] = audio_path
                    self._send({
                        "type": "chunk",
                        "index": i,
                        "text": fragment,
                        "audio_url": f"/audio/chunk/{i}",
                        "phonemes": [[ph, round(t, 4)] for ph, t in chunk_phonemes],
                    })
                self._send({"type": "tts_done"})
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


