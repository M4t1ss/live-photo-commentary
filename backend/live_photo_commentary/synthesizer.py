import io
import wave
from abc import ABC, abstractmethod
from typing import Generator

import numpy as np


type Timings = list[tuple[str, float]]
# (surface, cstart, cend, tstart, tend) with cstart/cend relative to the chunk text
type WordTiming = tuple[str, int, int, float, float]
# (audio, clean_text, phoneme_timings, word_timings|None, tag_timings)
type SynthResult = tuple[np.ndarray, str, Timings, list[WordTiming] | None, Timings]

class Synthesizer(ABC):
    def __new__(cls, **kwargs):
        """Factory method that returns the appropriate synthesizer implementation."""
        engine = kwargs.pop('engine', 'kokoro')
        if cls is not Synthesizer:
            # Direct instantiation of subclass
            return super().__new__(cls)

        # Factory logic for Synthesizer instantiation - delegate to specific implementations
        if engine == 'kokoro':
            from .synthesizers.kokoro import KokoroSynthesizer
            return KokoroSynthesizer(**kwargs)
        else:
            raise ValueError(
                f"Unsupported engine: {engine}. "
                f"Supported engines: kokoro"
            )

    def __init__(self, **kwargs):
        pass

    def __call__(self, text) -> Generator[SynthResult, None, None]:
        yield self.synthesize(text)

    @abstractmethod
    def synthesize(self, text) -> SynthResult:
        """Synthesize text to audio. Should yield audio chunks."""
        pass

    @abstractmethod
    def sample_rate(self) -> int:
        """Return the sample rate of the synthesized audio."""
        pass

    def to_wav_bytes(self, audio: np.ndarray) -> bytes:
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate())
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()
