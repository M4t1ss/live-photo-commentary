import torch

from ..local_describer import LocalDescriber, image_to_data_uri


class Gemma3LocalDescriber(LocalDescriber):
    def _create_model(self, quantization_config, attn_implementation, cuda_available):
        if "Qwen" in self.model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_kwargs = {
                "device_map": "cuda" if cuda_available and not self.device_param else None,
                "trust_remote_code": True,
                "quantization_config": quantization_config,
                "torch_dtype": "auto",
            } | self.model_kwargs
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
        return super()._create_model(quantization_config, attn_implementation, cuda_available)

    def build_messages(self, user_prompt, images=None, system_prompt=None):
        user_prompt = user_prompt.replace("<|image_1|>", "first image").replace("<|image_2|>", "second image")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({
            "role": "user",
            "content": [
                *[{"type": "image", "url": image_to_data_uri(image)} for image in (images or [])],
                {"type": "text", "text": user_prompt},
            ],
        })
        return messages

    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        messages = self.build_messages(user_prompt, images, system_prompt)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generate_ids = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                **self.generation_args,
            )

        return self.tokenizer.decode(
            generate_ids[:, input_len:][0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
