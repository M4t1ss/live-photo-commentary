import pytest
from unittest.mock import patch, MagicMock, ANY
from PIL import Image
from live_photo_commentary.describer import Describer


class Phi3VProvider:
    model_id = "microsoft/Phi-3.5-vision-instruct"

    @staticmethod
    def wire_fake_model(MockModel, MockProcessor):
        # Mock model parameters - return infinite iterator to avoid StopIteration
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_param.dtype = "float32"
        from itertools import repeat

        mock_model_instance = MagicMock()
        mock_generated = MagicMock()
        mock_generated.__getitem__ = lambda self, key: MagicMock()
        mock_model_instance.generate.return_value = mock_generated
        mock_model_instance.parameters = lambda: repeat(mock_param)
        # Ensure .to() returns self with same parameters() method
        mock_model_instance.to.return_value = mock_model_instance
        MockModel.from_pretrained.return_value = mock_model_instance

        mock_processor_instance = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, key: MagicMock(shape=(1, 10))
        mock_inputs.to.return_value = mock_inputs
        mock_processor_instance.return_value = mock_inputs
        mock_processor_instance.batch_decode.return_value = ["RESULT"]
        mock_processor_instance.tokenizer = MagicMock()
        mock_processor_instance.tokenizer.apply_chat_template.return_value = "TEMPLATED_PROMPT"
        mock_processor_instance.tokenizer.eos_token_id = 0
        MockProcessor.from_pretrained.return_value = mock_processor_instance

        return mock_model_instance, mock_processor_instance

    @staticmethod
    def assert_call(mock_processor, prompt, system_prompt, num_images=0):
        # Check that apply_chat_template was called with correct messages
        args, kwargs = mock_processor.tokenizer.apply_chat_template.call_args
        messages = args[0]

        if system_prompt:
            assert messages == [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ANY}
            ]
        else:
            assert messages == [
                {"role": "user", "content": ANY}
            ]


class Phi4MMProvider:
    model_id = "microsoft/Phi-4"

    @staticmethod
    def wire_fake_model(MockModel, MockProcessor):
        # Mock model parameters - return infinite iterator to avoid StopIteration
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_param.dtype = "float32"
        from itertools import repeat

        mock_model_instance = MagicMock()
        mock_generated = MagicMock()
        mock_generated.__getitem__ = lambda self, key: MagicMock()
        mock_model_instance.generate.return_value = mock_generated
        mock_model_instance.parameters = lambda: repeat(mock_param)
        mock_model_instance.to.return_value = mock_model_instance
        MockModel.from_pretrained.return_value = mock_model_instance

        mock_processor_instance = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, key: MagicMock(shape=(1, 10))
        mock_inputs.to.return_value = mock_inputs
        mock_processor_instance.return_value = mock_inputs
        mock_processor_instance.batch_decode.return_value = ["RESULT"]
        mock_processor_instance.tokenizer = MagicMock()
        mock_processor_instance.tokenizer.eos_token_id = 0
        MockProcessor.from_pretrained.return_value = mock_processor_instance

        return mock_model_instance, mock_processor_instance

    @staticmethod
    def assert_call(mock_processor, prompt, system_prompt, num_images=0):
        # Check that the processor was called with a prompt containing system prompt if provided
        args, kwargs = mock_processor.call_args
        prompt_text = args[0] if args else kwargs.get('text', '')

        if system_prompt:
            assert f"<|system|>{system_prompt}<|end|>" in prompt_text
        else:
            assert "<|system|>" not in prompt_text


class Gemma3Provider:
    model_id = "google/gemma-2-9b-it-vision"

    @staticmethod
    def wire_fake_model(MockModel, MockProcessor):
        # Mock model parameters - return infinite iterator to avoid StopIteration
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_param.dtype = "float32"
        from itertools import repeat

        mock_model_instance = MagicMock()
        mock_generated = MagicMock()
        mock_generated.__getitem__ = lambda self, key: MagicMock()
        mock_model_instance.generate.return_value = mock_generated
        mock_model_instance.parameters = lambda: repeat(mock_param)
        mock_model_instance.to.return_value = mock_model_instance
        MockModel.from_pretrained.return_value = mock_model_instance

        mock_processor_instance = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, key: MagicMock(shape=(1, 10))
        mock_inputs.to.return_value = mock_inputs
        mock_processor_instance.apply_chat_template.return_value = mock_inputs
        mock_processor_instance.tokenizer = MagicMock()
        mock_processor_instance.tokenizer.eos_token_id = 0
        mock_processor_instance.tokenizer.decode.return_value = "RESULT"
        MockProcessor.from_pretrained.return_value = mock_processor_instance

        return mock_model_instance, mock_processor_instance

    @staticmethod
    def assert_call(mock_processor, prompt, system_prompt, num_images=0):
        # Check that apply_chat_template was called with correct messages
        args, kwargs = mock_processor.apply_chat_template.call_args
        messages = args[0]

        if system_prompt:
            assert messages == [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": ANY}
            ]
        else:
            assert messages == [
                {"role": "user", "content": ANY}
            ]


@pytest.mark.parametrize("num_images", [0, 1, 2])
@pytest.mark.parametrize("ProviderClass", [Phi3VProvider, Phi4MMProvider, Gemma3Provider])
def test_prompt_model_with_system_prompt(ProviderClass, num_images):
    with patch("live_photo_commentary.local_describer.AutoModelForCausalLM") as MockModel, \
         patch("live_photo_commentary.local_describer.AutoProcessor") as MockProcessor:

        mock_model, mock_processor = ProviderClass.wire_fake_model(MockModel, MockProcessor)

        describer = Describer(
            local=True,
            model_id=ProviderClass.model_id,
            system_prompt="SYSTEM_PROMPT",
        )

        images = None
        if num_images > 0:
            img = Image.new("RGB", (100, 100), (255, 255, 255))
            images = [img] * num_images

        result = describer.prompt_model("PROMPT", images=images, system_prompt="SYSTEM_PROMPT")
        assert result == "RESULT"

        ProviderClass.assert_call(mock_processor, "PROMPT", "SYSTEM_PROMPT", num_images=num_images)


@pytest.mark.parametrize("num_images", [0, 1, 2])
@pytest.mark.parametrize("ProviderClass", [Phi3VProvider, Phi4MMProvider, Gemma3Provider])
def test_prompt_model_without_system_prompt(ProviderClass, num_images):
    with patch("live_photo_commentary.local_describer.AutoModelForCausalLM") as MockModel, \
         patch("live_photo_commentary.local_describer.AutoProcessor") as MockProcessor:

        mock_model, mock_processor = ProviderClass.wire_fake_model(MockModel, MockProcessor)

        describer = Describer(
            local=True,
            model_id=ProviderClass.model_id,
            system_prompt="SYSTEM_PROMPT",
        )

        images = None
        if num_images > 0:
            img = Image.new("RGB", (100, 100), (255, 255, 255))
            images = [img] * num_images

        result = describer.prompt_model("PROMPT", images=images, system_prompt=None)
        assert result == "RESULT"

        ProviderClass.assert_call(mock_processor, "PROMPT", None, num_images=num_images)
