"""Gemini API describer implementation."""

from google import genai
from google.genai import types as genai_types

from ..remote_describer import RemoteDescriber
from ..describer import image_to_bytes


class GeminiDescriber(RemoteDescriber):
    """Describer for Google Gemini API."""

    def get_default_model(self):
        """Return the default model ID for Gemini."""
        return "gemini-2.0-flash-exp"

    def _setup_client(self):
        """Setup the Gemini API client."""
        self.client = genai.Client(api_key=self.api_key)

    def build_messages(self, user_prompt, images=None, system_prompt=None):
        """Build content list for Gemini API."""
        if not images:
            images = []

        # Create the prompt with text and multiple images
        contents: genai_types.ContentListUnion = [
            user_prompt,
            *[
                genai_types.Part.from_bytes(data=image_to_bytes(image), mime_type='image/png')
                for image in images
            ]
        ]

        config = genai_types.GenerateContentConfig()
        if system_prompt:
            config.system_instruction = system_prompt

        return contents, config

    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        contents, config = self.build_messages(user_prompt, images, system_prompt)

        response = self.client.models.generate_content(
            model=self.model_id,
            config=config,
            contents=contents,
        )
        return response.text
