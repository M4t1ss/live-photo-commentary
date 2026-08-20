import base64

from openai import OpenAI

from ..remote_describer import RemoteDescriber
from ..describer import image_to_bytes


def image_to_base64_url(image):
    image_base64 = base64.b64encode(image_to_bytes(image)).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


class OpenAIDescriber(RemoteDescriber):
    def get_default_model(self):
        return "gpt-4o"

    def _setup_client(self):
        self.client = OpenAI(
            api_key=self.api_key or "none",
            base_url=self.base_url or None,
        )

    def _image_content(self, image):
        return {"type": "image_url", "image_url": {"url": image_to_base64_url(image)}}

    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        messages = self.build_messages(user_prompt, images, system_prompt)
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content
