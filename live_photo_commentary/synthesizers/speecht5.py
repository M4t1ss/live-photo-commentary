import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

from ..synthesizer import Synthesizer


class SpeechT5Synthesizer(Synthesizer):
    def __init__(self, model_id="microsoft/speecht5_tts", vocoder_id="microsoft/speecht5_hifigan", 
                 speaker_dataset="Matthijs/cmu-arctic-xvectors", speaker_idx=7306, **kwargs):
        super().__init__(**kwargs)
        
        self.model_id = model_id
        self.vocoder_id = vocoder_id
        
        # Load processor, model, and vocoder
        self.processor = SpeechT5Processor.from_pretrained(model_id)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_id)
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_id)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.vocoder = self.vocoder.to(self.device)
        
        # Load speaker embeddings
        embeddings_dataset = load_dataset(speaker_dataset, split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[speaker_idx]["xvector"]).unsqueeze(0).to(self.device)

    def synthesize(self, text):
        """Synthesize text to audio. Yields the complete waveform as a single chunk."""
        # Prepare text input
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        
        # Generate speech
        with torch.no_grad():
            speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
        
        # Move to CPU and convert to numpy
        waveform = speech.cpu().numpy()
        
        # Yield as (graphemes, phonemes, audio) tuple to match interface
        # Graphemes are the input text, phonemes are None for SpeechT5
        yield text, None, waveform

    def sample_rate(self):
        """Return the sample rate of the synthesized audio."""
        return 16000


if __name__ == '__main__':
    import sounddevice as sd
    synthesizer = SpeechT5Synthesizer()
    text = "Hello, world! This is a test of the SpeechT5 synthesizer."
    for gs, ps, audio in synthesizer(text):
        sd.play(audio, samplerate=synthesizer.sample_rate())
        sd.wait()