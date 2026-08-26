from abc import ABC, abstractmethod


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
        elif engine == 'vits':
            from .synthesizers.vits import VitsSynthesizer
            return VitsSynthesizer(**kwargs)
        elif engine == 'speecht5':
            from .synthesizers.speecht5 import SpeechT5Synthesizer
            return SpeechT5Synthesizer(**kwargs)
        else:
            raise ValueError(
                f"Unsupported engine: {engine}. "
                f"Supported engines: kokoro, vits, speecht5"
            )

    def __init__(self, max_chunk=None, **kwargs):
        self.max_chunk = max_chunk
        if max_chunk is not None:
            from semantic_text_splitter import TextSplitter
            self.text_splitter = TextSplitter(capacity=max_chunk)
        else:
            self.text_splitter = None

    def __call__(self, text):
        """Synthesize text to audio. Handles text splitting if max_chunk is set."""
        if self.text_splitter is not None:
            # Split text into chunks and synthesize each
            chunks = list(self.text_splitter.chunks(text))
            for chunk in chunks:
                yield from self.synthesize(chunk)
        else:
            # No splitting, synthesize the full text
            yield from self.synthesize(text)

    @abstractmethod
    def synthesize(self, text):
        """Synthesize text to audio. Should yield audio chunks."""
        pass

    @abstractmethod
    def sample_rate(self):
        """Return the sample rate of the synthesized audio."""
        pass


if __name__ == '__main__':
    import sounddevice as sd
    synthesizer = Synthesizer(engine='kokoro')
    text = "Hello, world!"
    for gs, ps, audio in synthesizer(text):
        sd.play(audio, samplerate=synthesizer.sample_rate())
        sd.wait()
