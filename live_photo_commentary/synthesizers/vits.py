import torch
from transformers import VitsModel, AutoTokenizer

from ..synthesizer import Synthesizer


class VitsSynthesizer(Synthesizer):
    def __init__(self, model_id="facebook/mms-tts-eng", **kwargs):
        super().__init__(**kwargs)
        
        self.model_id = model_id
        self.model = VitsModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        
        # Set seed for consistent output (MMS-TTS has stochastic components)
        torch.manual_seed(42)

    def synthesize(self, text):
        """Synthesize text to audio. Yields the complete waveform as a single chunk."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model(**inputs).waveform
        
        # Move to CPU and convert to numpy for consistency with other synthesizers
        waveform = output.cpu().float().numpy().squeeze()
        
        # Yield as (graphemes, phonemes, audio) tuple to match Kokoro interface
        # Graphemes are the input text, phonemes are None for VITS
        yield text, None, waveform

    def sample_rate(self):
        """Return the sample rate of the synthesized audio."""
        return self.model.config.sampling_rate


if __name__ == '__main__':
    import sounddevice as sd
    synthesizer = VitsSynthesizer()
    text = "Hello, world! This is a test of the MMS TTS synthesizer."
    for gs, ps, audio in synthesizer(text):
        sd.play(audio, samplerate=synthesizer.sample_rate())
        sd.wait()