"""Phi-4MM LocalDescriber implementation."""

from ..local_describer import LocalDescriber


class Phi4MMLocalDescriber(LocalDescriber):
    """LocalDescriber for Phi-4MM models."""
    
    def prompt_model(self, user_prompt, images=None) -> str | None:
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