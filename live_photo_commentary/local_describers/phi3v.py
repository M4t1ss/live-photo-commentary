"""Phi-3V LocalDescriber implementation."""

from ..local_describer import LocalDescriber


class Phi3VLocalDescriber(LocalDescriber):
    """LocalDescriber for Phi-3V models."""

    def build_messages(self, user_prompt, images=None, system_prompt=None):
        """Build chat messages for Phi-3V models."""
        if not images:
            images = []
        placeholder = ''.join(f"<|image_{ix + 1}|>\n" for ix, _ in enumerate(images))
        user_prompt = placeholder + user_prompt.replace("<|image_1|>", "first image").replace("<|image_2|>", "second image")

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        return messages

    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        messages = self.build_messages(user_prompt, images, system_prompt)

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(prompt, images or None, return_tensors="pt").to(self.device)
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
