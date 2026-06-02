from abc import ABC, abstractmethod
from typing import Generator

import numpy as np


type SynthResult = tuple[np.ndarray, str, list[tuple[str, float]]]


class Synthesizer(ABC):

    def __new__(cls, **kwargs):
        engine = kwargs.pop('engine', 'kokoro')
        if cls is not Synthesizer:
            return super().__new__(cls)

        if engine == 'kokoro':
            from .synthesizers.kokoro import KokoroSynthesizer
            return KokoroSynthesizer(**kwargs)
        else:
            raise ValueError(
                f"Unsupported engine: {engine}. "
                f"Supported engines: kokoro"
            )

    def __init__(self, text_splitter=None, **kwargs):
        self.text_splitter = text_splitter

    def __call__(self, text) -> Generator[SynthResult, None, None]:
        if self.text_splitter is not None:
            for chunk in self.text_splitter.chunks(text):
                yield self.synthesize(chunk)
        else:
            yield self.synthesize(text)

    @abstractmethod
    def synthesize(self, text) -> SynthResult:
        pass

    @abstractmethod
    def sample_rate(self) -> int:
        pass
