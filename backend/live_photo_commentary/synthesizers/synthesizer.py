from abc import ABC, abstractmethod
from typing import Generator

import numpy as np


type Timings = list[tuple[str, float]]
type SynthResult = tuple[np.ndarray, str, Timings, Timings]

class Synthesizer(ABC):

    def __new__(cls, **kwargs):
        """Factory method that returns the appropriate synthesizer implementation."""
        engine = kwargs.pop('engine', 'kokoro')
        if cls is not Synthesizer:
            # Direct instantiation of subclass
            return super().__new__(cls)
        
        # Factory logic for Synthesizer instantiation - delegate to specific implementations
        if engine == 'kokoro':
            from .kokoro import KokoroSynthesizer
            return KokoroSynthesizer(**kwargs)
        else:
            raise ValueError(
                f"Unsupported engine: {engine}. "
                f"Supported engines: kokoro, vits, speecht5"
            )

    def __init__(self, text_splitter=None, **kwargs):
        if text_splitter is not None:
            self.text_splitter = text_splitter
        else:
            self.text_splitter = None

    def __call__(self, text) -> Generator[SynthResult, None, None]:
        """Synthesize text to audio. Handles text splitting if max_chunk is set."""
        if self.text_splitter is not None:
            # Split text into chunks and synthesize each
            chunks = list(self.text_splitter.chunks(text))
            for chunk in chunks:
                yield self.synthesize(chunk)
        else:
            # No splitting, synthesize the full text
            yield self.synthesize(text)

    @abstractmethod
    def synthesize(self, text) -> SynthResult:
        """Synthesize text to audio. Should yield audio chunks."""
        pass

    @abstractmethod
    def sample_rate(self) -> int:
        """Return the sample rate of the synthesized audio."""
        pass
