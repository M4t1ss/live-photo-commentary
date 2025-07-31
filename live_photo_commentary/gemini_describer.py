import os

from google import genai
from google.genai import types as genai_types



from .describer import (
    Describer,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_ENDING,
    DEFAULT_FIRST_PROMPT,
    DEFAULT_PROMPT,
    DEFAULT_HISTORY_PROMPT,
    DEFAULT_COMPACT_PROMPT,
)


class GeminiDescriber(Describer):
    def __init__(self,
             gemini_api_key=None,
             system_prompt=DEFAULT_SYSTEM_PROMPT,
             ending=DEFAULT_ENDING,
             first_prompt=DEFAULT_FIRST_PROMPT,
             prompt=DEFAULT_PROMPT,
             history_prompt=DEFAULT_HISTORY_PROMPT,
             compact_prompt=DEFAULT_COMPACT_PROMPT,
             max_history_size=False,
    ):
        super().__init__(
             system_prompt=system_prompt,
             ending=ending,
             first_prompt=first_prompt,
             prompt=prompt,
             history_prompt=history_prompt,
             compact_prompt=compact_prompt,
             max_history_size=max_history_size,
        )

        if not gemini_api_key:
            gemini_api_key = os.environ["GEMINI_API_KEY"]

        self.client = genai.Client(api_key=gemini_api_key)


    def prompt_model(self, user_prompt, images=None) -> str | None:
        if not images:
            images = []

        # Create the prompt with text and multiple images (if appropriate)
        contents: genai_types.ContentListUnion = [
            genai_types.Part.from_bytes(data=image, mime_type='image/png') for image in images
        ]
        contents.insert(0, user_prompt)

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            config=genai_types.GenerateContentConfig(system_instruction = self.system_prompt),
            contents=contents,
        )
        return response.text



if __name__ == '__main__':
    from PIL import ImageGrab
    screenshot = ImageGrab.grab()
    describer = GeminiDescriber()
    description = describer(screenshot)
    print(description)
