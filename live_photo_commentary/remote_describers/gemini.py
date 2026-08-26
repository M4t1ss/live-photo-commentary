"""Gemini API describer implementation."""

from google import genai
from google.genai import types as genai_types
from google.genai import errors  # Import errors to catch 503
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..remote_describer import RemoteDescriber
from ..describer import image_to_bytes
import logging

class NoThoughtSignatureWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Blocks the non-text parts warning message from printing
        if "there are non-text parts in the response" in record.getMessage():
            return False
        return True

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

    # Automatically retry only on ServerError (503). 
    # Starts waiting 2s, increases exponentially, up to 5 times.
    @retry(
        retry=retry_if_exception_type(errors.ServerError),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def _call_api_with_retry(self, model, config, contents):
        """Helper method wrapped with retry logic."""
        return self.client.models.generate_content(
            model=model,
            config=config,
            contents=contents,
        )

    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        contents, config = self.build_messages(user_prompt, images, system_prompt)
        logging.getLogger("google_genai.types").addFilter(NoThoughtSignatureWarning())

        # Call the retry-protected helper instead of the raw client
        response = self._call_api_with_retry(
            model=self.model_id,
            config=config,
            contents=contents,
        )
        return response.text
