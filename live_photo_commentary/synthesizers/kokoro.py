import warnings

from kokoro import KPipeline
import torch

from ..synthesizer import Synthesizer


class KokoroSynthesizer(Synthesizer):
    def __init__(self, voice=None, **kwargs):
        super().__init__(**kwargs)
        
        if voice:
            self.voice = voice
        else:
            voice_tensor1 = torch.load('voices/af_nicole.pt', weights_only=True)
            voice_tensor2 = torch.load('voices/jf_alpha.pt', weights_only=True)
            t = 0.3
            self.voice = (1 - t) * voice_tensor1 + t * voice_tensor2

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 
                category=UserWarning,
                module='torch.nn.modules.rnn',
            )
            warnings.filterwarnings('ignore',
                category=UserWarning,
                module='torch.nn.utils.weight_norm',
            )
            self.pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

    def synthesize(self, text):
        yield from self.pipeline(text, voice=self.voice, speed=1, split_pattern=r'\n+')

    def sample_rate(self):
        return 24000


if __name__ == '__main__':
    import sounddevice as sd
    synthesizer = KokoroSynthesizer()
    text = "Hello, world!"
    for gs, ps, audio in synthesizer(text):
        sd.play(audio, samplerate=synthesizer.sample_rate())
        sd.wait()
