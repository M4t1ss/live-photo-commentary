import importlib.util
from abc import abstractmethod
from io import BytesIO
import base64

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
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
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


class LocalDescriber(Describer):
    uses_processor = True
    processor_extra_kwargs: dict = {}

    def _display_name(self) -> str:
        """Return a clean HF-style model name even when model_id is a local path."""
        from pathlib import Path
        p = Path(self.model_id)
        # Local paths contain a separator; convert google--gemma-4-E4B-it → google/gemma-4-E4B-it
        if p.is_absolute() or (p.parts and any(sep in self.model_id for sep in ('/', '\\'))):
            return p.name.replace('--', '/', 1)
        return self.model_id

    def _notify(self, message: str) -> None:
        print(f"[lpc] {message}", flush=True)
        if self._on_progress:
            self._on_progress({"type": "load_progress", "message": message})

    def _make_weight_tqdm(self):
        """Return a tqdm subclass that forwards weight-loading progress to _on_progress."""
        import tqdm as tqdm_lib
        on_progress = self._on_progress

        class _WeightTqdm(tqdm_lib.tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._last_n = 0

            def update(self, n=1):
                super().update(n)
                if not self.total:
                    return
                if self.n - self._last_n >= max(self.total * 0.01, 1) or self.n >= self.total:
                    self._last_n = self.n
                    pct = round(self.n / self.total * 100)
                    on_progress({"type": "load_progress",
                                 "message": f"{self.desc or 'Loading weights'}  {pct}%"})

        return _WeightTqdm

    def __new__(cls, model_id="microsoft/Phi-4-multimodal-instruct", **kwargs):
        if cls is not LocalDescriber:
            return super().__new__(cls)

        if "FastVLM" in model_id:
            from .local_describers.fastvlm import FastVLMLocalDescriber
            instance = super(LocalDescriber, FastVLMLocalDescriber).__new__(FastVLMLocalDescriber)
        elif "Qwen" in model_id or "gemma" in model_id.lower():
            from .local_describers.gemma3 import Gemma3LocalDescriber
            instance = super(LocalDescriber, Gemma3LocalDescriber).__new__(Gemma3LocalDescriber)
        elif "Phi-4" in model_id:
            from .local_describers.phi4mm import Phi4MMLocalDescriber
            instance = super(LocalDescriber, Phi4MMLocalDescriber).__new__(Phi4MMLocalDescriber)
        else:
            raise ValueError(
                f"Unsupported model: {model_id}. "
                f"Supported models: FastVLM, Phi-4, Gemma, Qwen"
            )

        return instance

    def __init__(self,
                 system_prompt=DEFAULT_SYSTEM_PROMPT,
                 ending=DEFAULT_ENDING,
                 first_prompt=DEFAULT_FIRST_PROMPT,
                 prompt=DEFAULT_PROMPT,
                 history_prompt=DEFAULT_HISTORY_PROMPT,
                 compact_prompt=DEFAULT_COMPACT_PROMPT,
                 max_history_size=False,
                 min_history_size=False,
                 model_id="microsoft/Phi-4-multimodal-instruct",
                 device=None,
                 processor_kwargs=None,
                 tokenizer_kwargs=None,
                 model_kwargs=None,
                 generation_kwargs=None,
                 on_progress=None,
                 **kwargs,
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
        self.model_id = model_id
        self.device_param = device
        self.processor_kwargs = processor_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.generation_kwargs = generation_kwargs or {}
        self._on_progress = on_progress
        self._setup_model()

    def _setup_model(self):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True) if importlib.util.find_spec('bitsandbytes') else None
        if importlib.util.find_spec('flash_attn'):
            attn_implementation = 'flash_attention_2'
        else:
            attn_implementation = 'sdpa'  # PyTorch built-in fused attention; much faster than eager
        cuda_available = torch.cuda.is_available()

        self._notify(f"Loading {self._display_name()} (cuda={cuda_available}, quant={quantization_config is not None}, attn={attn_implementation})")

        # Patch tqdm in transformers so weight-loading progress reaches the frontend.
        if self._on_progress:
            import importlib as _il
            _WeightTqdm = self._make_weight_tqdm()
            _patches = []
            for mod_name, attr in [('tqdm', 'tqdm'), ('tqdm.auto', 'tqdm'),
                                    ('transformers.modeling_utils', 'tqdm')]:
                try:
                    mod = _il.import_module(mod_name)
                    if hasattr(mod, attr):
                        _patches.append((mod, attr, getattr(mod, attr)))
                        setattr(mod, attr, _WeightTqdm)
                except ImportError:
                    pass

        try:
            self.model = self._create_model(quantization_config, attn_implementation, cuda_available)
        finally:
            if self._on_progress:
                for mod, attr, orig in _patches:
                    setattr(mod, attr, orig)

        self._notify("Model loaded")

        if not cuda_available:
            device = torch.device(self.device_param or ("mps" if torch.backends.mps.is_available() else "cpu"))
            self._notify(f"Moving model to {device}")
            try:
                self.model = self.model.to(device)
            except TypeError:
                self.model = self.model.to(device, dtype=torch.float16)

        if self.uses_processor:
            self._notify("Loading processor")
            processor_kwargs = {"trust_remote_code": True} | self.processor_extra_kwargs | self.processor_kwargs
            self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
        try:
            self.tokenizer = self.processor.tokenizer
        except AttributeError:
            tokenizer_kwargs = {"trust_remote_code": True} | self.tokenizer_kwargs
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)

        self.generation_args = {
            "max_new_tokens": 200,
            "temperature": 0.2,
            "do_sample": True,
        } | self.generation_kwargs

        model_params = next(self.model.parameters())
        self.device = model_params.device
        self.dtype = model_params.dtype
        self._notify(f"{self._display_name()} ready (device={self.device}, dtype={self.dtype})")

    def _create_model(self, quantization_config, attn_implementation, cuda_available):
        model_kwargs = {
            "device_map": "cuda" if cuda_available and not self.device_param else None,
            "trust_remote_code": True,
            "quantization_config": quantization_config,
            "torch_dtype": "auto",
            "_attn_implementation": attn_implementation,
        } | self.model_kwargs
        return AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

    @abstractmethod
    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        pass
