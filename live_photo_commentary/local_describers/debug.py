"""Debug LocalDescriber implementation for testing."""

import random

from ..local_describer import LocalDescriber
from ..describer import Describer


class DebugDescriber(LocalDescriber):
    """Debug describer that predictably generates placeholder text without loading a real model."""

    uses_processor = False

    def __init__(self, text='', model_id=None, local=None, **kwargs):
        # Skip LocalDescriber.__init__ and call Describer.__init__ directly
        Describer.__init__(self, **kwargs)
        self.text = text

    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        if callable(self.text):
            return self.text()
        else:
            return self.text
