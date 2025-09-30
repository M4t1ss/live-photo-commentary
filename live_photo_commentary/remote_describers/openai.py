"""OpenAI API describer implementation."""

import base64

from openai import OpenAI

from ..remote_describer import RemoteDescriber
from ..describer import image_to_bytes


def image_to_base64_url(image):
    """Convert PIL image to base64 data URL for OpenAI API."""
    image_bytes = image_to_bytes(image)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


class OpenAIDescriber(RemoteDescriber):
    """Describer for OpenAI API."""
    
    def get_default_model(self):
        """Return the default model ID for OpenAI."""
        return "gpt-4o"
    
    def _setup_client(self):
        """Setup the OpenAI API client."""
        self.client = OpenAI(api_key=self.api_key)
    
    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        if not images:
            images = []

        # Build the content array with text and images
        content = []
        
        # Add text content
        content.append({
            "type": "text",
            "text": user_prompt
        })
        
        # Add images
        for image in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_to_base64_url(image)
                }
            })

        # Build messages array
        if system_prompt:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
            ]
        else:
            messages = []
        messages.append(
            {
                "role": "user",
                "content": content
            }
        )

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content
