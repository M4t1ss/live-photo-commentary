import importlib.util
from io import BytesIO
import base64

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.pipelines import image_segmentation
from transformers.utils.quantization_config import BitsAndBytesConfig

from .describer import (
    Describer,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_ENDING,
    DEFAULT_FIRST_PROMPT,
    DEFAULT_PROMPT,
    DEFAULT_HISTORY_PROMPT,
    DEFAULT_COMPACT_PROMPT,
)



def image_to_data_uri(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    image_bytes = buffer.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Return as data URL
    return f"data:image/png;base64,{image_base64}"


class LocalDescriber(Describer):
    def __init__(self,
             system_prompt=DEFAULT_SYSTEM_PROMPT,
             ending=DEFAULT_ENDING,
             first_prompt=DEFAULT_FIRST_PROMPT,
             prompt=DEFAULT_PROMPT,
             history_prompt=DEFAULT_HISTORY_PROMPT,
             compact_prompt=DEFAULT_COMPACT_PROMPT,
             max_history_size=False,
             min_history_size=False,
             model_id="microsoft/Phi-3.5-vision-instruct",
             device=None,
    ):
        super().__init__(
             system_prompt=system_prompt,
             ending=ending,
             first_prompt=first_prompt,
             prompt=prompt,
             history_prompt=history_prompt,
             compact_prompt=compact_prompt,
             max_history_size=max_history_size,
             min_history_size=min_history_size,
        )

        quantization_config = BitsAndBytesConfig(load_in_4bit=True) if importlib.util.find_spec('bitsandbytes') else None

        attn_implementation = 'flash_attention_2' if importlib.util.find_spec('flash_attn') else 'eager'
        cuda_available = torch.cuda.is_available()

        if model_id == "Qwen/Qwen2.5-VL-3B-Instruct":
            from transformers import Qwen2_5_VLForConditionalGeneration

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map="cuda" if cuda_available and not device else None,
                trust_remote_code=True,
                quantization_config=quantization_config,
                torch_dtype="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cuda" if cuda_available and not device else None,
                trust_remote_code=True,
                quantization_config=quantization_config,
                torch_dtype="auto",
                _attn_implementation=attn_implementation
            )

        if not cuda_available:
            device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
            # Fallback dtype
            try:
                self.model = self.model.to(device)
            except TypeError:
                self.model = self.model.to(device, dtype=torch.float16)

        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=4
        )
        try:
            self.tokenizer = self.processor.tokenizer
        except AttributeError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )

        self.generation_args = {
            "max_new_tokens": 200,
            "temperature": 0.2,
            "do_sample": True,
        }

        self.device = next(self.model.parameters()).device

    def prompt_model(self, user_prompt, images=None) -> str | None:

        model_type = self.model.__class__.__name__
        if model_type == "Gemma3ForConditionalGeneration":
            return self.prompt_model_gemma3(user_prompt, images)
        elif model_type == "Phi3VForCausalLM":
            return self.prompt_model_phi3V(user_prompt, images)
        elif model_type == "Phi4MMForCausalLM":
            return self.prompt_model_phi4MM(user_prompt, images)
        elif model_type == "Qwen2_5_VLForConditionalGeneration":
            # no changes from Gemma 3 API
            return self.prompt_model_gemma3(user_prompt, images)
        else:
            raise NotImplementedError("Unsupported model")

    def prompt_model_phi3V(self, user_prompt, images=None):
        if not images:
            images = []
        placeholder = ''.join(f"<|image_{ix + 1}|>\n" for ix, _ in enumerate(images))
        user_prompt = placeholder + user_prompt.replace("<|image_1|>", "first image").replace("<|image_2|>", "second image")

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(prompt, images, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        # Avoid Phi bug on MPS
        if 'image_sizes' in inputs:
            inputs['image_sizes'] = inputs['image_sizes'].tolist()

        generate_ids = self.model.generate(**inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.generation_args
        )

        # remove input tokens
        generate_ids = generate_ids[:, input_len:]
        response_text = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response_text

    def prompt_model_phi4MM(self, user_prompt, images=None):
        if not images:
            images = []
        placeholder = ''.join(f"<|image_{ix + 1}|>\n" for ix, _ in enumerate(images))
        user_prompt = placeholder + user_prompt.replace("<|image_1|>", "first image").replace("<|image_2|>", "second image")
        prompt = (
            f"<|system|>{self.system_prompt}<|end|>"
            f"<|user|>{user_prompt}<|end|><|assistant|>"
        )

        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        # Avoid Phi bug on MPS
        if 'image_sizes' in inputs:
            inputs['image_sizes'] = inputs['image_sizes'].tolist()

        # XXX: Need this?
        # generation_config = GenerationConfig.from_pretrained(model_path)
        generate_ids = self.model.generate(**inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            num_logits_to_keep=1,
            # XXX: Need this?
            # generation_config=generation_config,
            **self.generation_args
        )

        # remove input tokens
        generate_ids = generate_ids[:, input_len:]
        response_text = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response_text

    def prompt_model_gemma3(self, user_prompt, images=None):
        user_prompt = user_prompt.replace("<|image_1|>", "first image").replace("<|image_2|>", "second image")

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_prompt,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    *[
                        {
                            "type": "image",
                            "url": image_to_data_uri(image),
                        }
                        for image in images or []
                    ],
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generate_ids = self.model.generate(**inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                **self.generation_args
            )

        generate_ids = generate_ids[:, input_len:][0]
        response_text = self.tokenizer.decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return response_text
