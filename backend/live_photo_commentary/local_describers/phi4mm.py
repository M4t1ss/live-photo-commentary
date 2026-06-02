from ..local_describer import LocalDescriber


class Phi4MMLocalDescriber(LocalDescriber):
    def build_messages(self, user_prompt, images=None, system_prompt=None):
        images = images or []
        placeholder = ''.join(f"<|image_{ix + 1}|>\n" for ix in range(len(images)))
        user_prompt = placeholder + user_prompt.replace("<|image_1|>", "first image").replace("<|image_2|>", "second image")
        system_fragment = f"<|system|>{system_prompt}<|end|>" if system_prompt else ""
        return f"{system_fragment}<|user|>{user_prompt}<|end|><|assistant|>"

    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        images = images or []
        prompt = self.build_messages(user_prompt, images, system_prompt)

        inputs = self.processor(text=prompt, images=images or None, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        if 'image_sizes' in inputs:
            inputs['image_sizes'] = inputs['image_sizes'].tolist()

        generate_ids = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            num_logits_to_keep=1,
            **self.generation_args,
        )

        return self.processor.batch_decode(
            generate_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
