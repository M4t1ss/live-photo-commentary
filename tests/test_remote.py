import pytest
from unittest.mock import patch, MagicMock, ANY
from PIL import Image
from live_photo_commentary.describer import Describer


class OpenAIProvider:
    patch_path = "live_photo_commentary.remote_describers.openai.OpenAI"

    @staticmethod
    def wire_fake_response(MockClient):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "RESULT"
        MockClient.return_value.chat.completions.create.return_value = resp
        return resp

    @staticmethod
    def assert_call(MockClient, prompt, system_prompt, num_images=0):
        args, kwargs = MockClient.return_value.chat.completions.create.call_args

        expected_messages = []
        if system_prompt:
            expected_messages.append({"role": "system", "content": system_prompt})

        expected_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ] + [
                {"type": "image_url", "image_url": {"url": ANY}}
            ] * num_images
        })

        assert kwargs == {
            "model": "MODEL",
            "messages": expected_messages,
            "max_tokens": ANY,
            "temperature": ANY,
        }


class GeminiProvider:
    patch_path = "live_photo_commentary.remote_describers.gemini.genai.Client"

    @staticmethod
    def wire_fake_response(MockClient):
        resp = MagicMock()
        resp.candidates = [MagicMock()]
        resp.text = "RESULT"
        MockClient.return_value.models.generate_content.return_value = resp
        return resp

    @staticmethod
    def assert_call(MockClient, prompt, system_prompt, num_images=0):
        args, kwargs = MockClient.return_value.models.generate_content.call_args
        assert kwargs == {
            "model": "MODEL",
            "config": ANY,
            "contents": [prompt] + [ANY] * num_images
        }
        if system_prompt:
            assert kwargs["config"].system_instruction == system_prompt
        else:
            assert not hasattr(kwargs["config"], "system_instruction") or kwargs["config"].system_instruction is None

@pytest.mark.parametrize("num_images", [0, 1, 2])
@pytest.mark.parametrize("ProviderClass", [OpenAIProvider, GeminiProvider])
def test_prompt_model_with_system_prompt(ProviderClass, num_images):
    with patch(ProviderClass.patch_path) as MockClient:
        # Wire provider-specific fake response
        fake_response = ProviderClass.wire_fake_response(MockClient)

        describer = Describer(
            local=False,
            provider=ProviderClass.__name__.replace("Provider", "").lower(),
            model_id="MODEL",
            system_prompt="SYSTEM_PROMPT",
        )

        images = None
        if num_images > 0:
            img = Image.new("RGB", (100, 100), (255, 255, 255))
            images = [img] * num_images

        result = describer.prompt_model("PROMPT", images=images, system_prompt="SYSTEM_PROMPT")
        assert result == "RESULT"

        ProviderClass.assert_call(MockClient, "PROMPT", "SYSTEM_PROMPT", num_images=num_images)


@pytest.mark.parametrize("num_images", [0, 1, 2])
@pytest.mark.parametrize("ProviderClass", [OpenAIProvider, GeminiProvider])
def test_prompt_model_without_system_prompt(ProviderClass, num_images):
    with patch(ProviderClass.patch_path) as MockClient:
        # Wire provider-specific fake response
        fake_response = ProviderClass.wire_fake_response(MockClient)

        describer = Describer(
            local=False,
            provider=ProviderClass.__name__.replace("Provider", "").lower(),
            model_id="MODEL",
            system_prompt="SYSTEM_PROMPT",
        )

        images = None
        if num_images > 0:
            img = Image.new("RGB", (100, 100), (255, 255, 255))
            images = [img] * num_images

        result = describer.prompt_model("PROMPT", images=images, system_prompt=None)
        assert result == "RESULT"

        ProviderClass.assert_call(MockClient, "PROMPT", None, num_images=num_images)

