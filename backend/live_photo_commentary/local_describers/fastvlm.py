import logging

import torch
from transformers import AutoModelForCausalLM

from ..local_describer import LocalDescriber

log = logging.getLogger(__name__)


class FastVLMLocalDescriber(LocalDescriber):
    uses_processor = False
    IMAGE_TOKEN_INDEX = -200

    def _create_model(self, quantization_config, attn_implementation, device):
        model_kwargs = {
            "device_map": "cuda" if device == "cuda" and not self.device_param else None,
            "trust_remote_code": True,
            "quantization_config": quantization_config,
            "torch_dtype": torch.float16 if device in ("cuda", "mps") else torch.float32,
            "attn_implementation": attn_implementation,
        } | self.model_kwargs
        log.info("model_kwargs: %s", model_kwargs)
        return AutoModelForCausalLM.from_pretrained(self.load_path, **model_kwargs)

    def build_messages(self, user_prompt, images=None, system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        images = images or []
        messages = self.build_messages(user_prompt, images, system_prompt)

        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        parts = rendered.split("<image>")
        input_ids_parts = []

        for i in range(len(images)):
            if parts[i]:
                text_ids = self.tokenizer(parts[i], return_tensors="pt", add_special_tokens=False).input_ids
                input_ids_parts.append(text_ids)
            input_ids_parts.append(torch.tensor([[self.IMAGE_TOKEN_INDEX]], dtype=torch.long))

        if parts[-1]:
            text_ids = self.tokenizer(parts[-1], return_tensors="pt", add_special_tokens=False).input_ids
            input_ids_parts.append(text_ids)

        input_ids = torch.cat(input_ids_parts, dim=1).to(self.device)

        images_rgb = [image.convert('RGB') for image in images]
        if images_rgb:
            images_tensor = self.model.get_vision_tower().image_processor(
                images=images_rgb, return_tensors="pt"
            )["pixel_values"].to(self.device, dtype=self.model.dtype)
        else:
            images_tensor = None

        attention_mask = torch.ones_like(input_ids, device=self.device)

        with torch.inference_mode():
            generate_ids = self.model.generate(
                inputs=input_ids,
                images=images_tensor,
                attention_mask=attention_mask,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.generation_args,
            )

        return self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
