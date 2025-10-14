"""Dummy LocalDescriber implementation for testing."""

import random
from dummy_text_generator import generate_sentence

from ..local_describer import LocalDescriber
from ..describer import Describer


class DummyDescriber(LocalDescriber):
    """Dummy describer that generates placeholder text without loading a real model."""

    uses_processor = False

    def __init__(self, min_sentences=1, max_sentences=None, model_id=None, local=None, **kwargs):
        # Skip LocalDescriber.__init__ and call Describer.__init__ directly
        Describer.__init__(self, **kwargs)
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences if max_sentences is not None else min_sentences

    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        num_sentences = random.randint(self.min_sentences, self.max_sentences)
        dummy_text = ' '.join(generate_sentence('en', None) for _ in range(num_sentences))
        return dummy_text
