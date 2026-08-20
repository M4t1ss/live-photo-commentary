from google import genai
from google.genai import types as genai_types

from ..remote_describer import RemoteDescriber
from ..describer import image_to_bytes, split_prompt_images


class GeminiDescriber(RemoteDescriber):
    def get_default_model(self):
        return "gemini-2.0-flash-exp"

    def _setup_client(self):
        self.client = genai.Client(api_key=self.api_key)

    def build_messages(self, user_prompt, images=None, system_prompt=None):
        images = images or []
        config = genai_types.GenerateContentConfig()
        if system_prompt:
            config.system_instruction = system_prompt

        segments = split_prompt_images(user_prompt)
        if not any(kind == 'image' for kind, _ in segments):
            contents = [user_prompt,
                        *[genai_types.Part.from_bytes(data=image_to_bytes(img), mime_type='image/png')
                          for img in images]]
        else:
            contents = []
            for kind, val in segments:
                if kind == 'text':
                    contents.append(val)
                elif val < len(images):
                    contents.append(genai_types.Part.from_bytes(
                        data=image_to_bytes(images[val]), mime_type='image/png'))

        return contents, config

    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        contents, config = self.build_messages(user_prompt, images, system_prompt)
        response = self.client.models.generate_content(
            model=self.model_id,
            config=config,
            contents=contents,
        )
        return response.text
