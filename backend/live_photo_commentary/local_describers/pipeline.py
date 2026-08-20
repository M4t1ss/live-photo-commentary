import logging

from ..local_describer import LocalDescriber, image_to_data_uri

log = logging.getLogger(__name__)


class PipelineLocalDescriber(LocalDescriber):
    def _setup_model(self):
        from transformers import pipeline

        cuda_available, device = self._select_device()
        quantization_config = self._select_quant(cuda_available)
        attn_implementation = self._select_attn()

        self._notify(f"Loading {self._display_name()} via pipeline (device={device}, quant={quantization_config is not None}, attn={attn_implementation})")

        pipeline_kwargs = {
            "model_kwargs": {
                "quantization_config": quantization_config,
                "_attn_implementation": attn_implementation,
                "torch_dtype": "auto",
                **self.model_kwargs,
            },
            "trust_remote_code": True,
        }
        if cuda_available and not self.device_param:
            # BNB 4-bit requires all layers on CUDA; "auto" plans layout from the
            # non-quantized size so a 12B model looks like ~24 GB and accelerate
            # spills layers to CPU, which BNB then rejects (broke in accelerate
            # 1.14.0). Use "cuda" so BNB quantizes in place; keep "auto" only
            # when not quantizing, where multi-GPU spread is desirable.
            pipeline_kwargs["device_map"] = "cuda" if quantization_config else "auto"
        else:
            pipeline_kwargs["device"] = self.device_param or device

        with self._tqdm_patched():
            self.pipe = pipeline("image-text-to-text", model=self.load_path, **pipeline_kwargs)

        self._notify("Model loaded")
        self._maybe_enable_xformers(self.pipe.model, attn_implementation)

        # Expose model/processor/tokenizer for parent infrastructure (_notify dtype, tqdm patch).
        self.model = self.pipe.model
        self.processor = self.pipe.processor
        try:
            self.tokenizer = self.pipe.processor.tokenizer
        except AttributeError:
            self.tokenizer = self.pipe.tokenizer

        # processor_kwargs: pipeline has no forwarding path, so patch post-hoc.
        # Processor attributes (e.g. max_pixels for Qwen) are plain instance attrs
        # set by from_pretrained, so setting them here has the same effect.
        if self.processor_kwargs:
            image_proc = getattr(self.processor, 'image_processor', self.processor)
            for k, v in self.processor_kwargs.items():
                setattr(image_proc, k, v)

        self.generation_args = self.default_generation_args | self.generation_kwargs

        model_params = next(self.model.parameters())
        self.device = model_params.device
        self.dtype = model_params.dtype
        self._notify(f"{self._display_name()} ready (device={self.device}, dtype={self.dtype})")

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

        result = self.pipe(
            messages,
            return_full_text=False,
            generate_kwargs=self.generation_args,
        )
        text = result[-1]['generated_text']

        # Some models (e.g. Gemma) leak their EOS token strings into decoded output.
        eos_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        eos_strings = [
            self.tokenizer.decode([tid], skip_special_tokens=False)
            for tid in (eos_ids or [])
        ]
        while True:
            prev = text
            for suffix in eos_strings:
                text = text.removesuffix(suffix)
            if text == prev:
                break

        return text.strip()
