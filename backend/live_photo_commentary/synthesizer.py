import io
import wave
from abc import ABC, abstractmethod
from typing import Generator

import numpy as np


type Timings = list[tuple[str, float]]
type SynthResult = tuple[np.ndarray, str, Timings, Timings, list[str]]

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
