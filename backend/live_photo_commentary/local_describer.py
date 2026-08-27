import contextlib
import importlib.util
from abc import abstractmethod
from io import BytesIO
import base64

# torch may have been installed (or reinstalled) moments before this process
# started (CUDA venv setup via restart_backend); see _wait_for_torch.py.
from ._wait_for_torch import wait_for_torch
wait_for_torch()

import logging

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

log = logging.getLogger(__name__)

from .describer import (
    Describer,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_ENDING,
    DEFAULT_FIRST_PROMPT,
    DEFAULT_PROMPT,
    DEFAULT_HISTORY_PROMPT,
    DEFAULT_COMPACT_PROMPT,
)


def _flash_attn2_compatible(load_path):
    """Return True only if every attention layer has head_dim ≤ 256 (flash_attn 2 limit).

    Models with heterogeneous per-layer configs (e.g. Gemma 4) raise
    AmbiguousGlobalPerLayerAttributeError on head_dim access — treat that as
    incompatible. Models that simply don't define head_dim (e.g. Qwen2.5-VL)
    get it computed from hidden_size / num_attention_heads.
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(load_path, trust_remote_code=True)
        tc = getattr(cfg, "text_config", cfg)
        try:
            hd = tc.head_dim
        except AttributeError:
            hd = None  # not defined; compute below
        except Exception:
            return False  # AmbiguousGlobalPerLayerAttributeError — heterogeneous per-layer config
        if hd is None:
            hd = getattr(tc, "hidden_size", 256) // max(getattr(tc, "num_attention_heads", 1), 1)
        return isinstance(hd, int) and hd <= 256
    except Exception:
        return True  # can't read config → assume ok


def image_to_data_uri(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


class LocalDescriber(Describer):
    uses_processor = True
    processor_extra_kwargs: dict = {}
    tokenizer_extra_kwargs: dict = {}
    default_generation_args: dict = {
        "max_new_tokens": 200,
        "max_length": None,  # suppress spurious warning when model config has max_length set
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 50,
    }

    def _display_name(self) -> str:
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
        elif "Phi-4" in model_id:
            from .local_describers.phi4mm import Phi4MMLocalDescriber
            instance = super(LocalDescriber, Phi4MMLocalDescriber).__new__(Phi4MMLocalDescriber)
        elif "InternVL" in model_id:
            from .local_describers.internvl import InternVLLocalDescriber
            instance = super(LocalDescriber, InternVLLocalDescriber).__new__(InternVLLocalDescriber)
        else:
            from .local_describers.pipeline import PipelineLocalDescriber
            instance = super(LocalDescriber, PipelineLocalDescriber).__new__(PipelineLocalDescriber)

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
                 load_path=None,
                 device=None,
                 processor_kwargs=None,
                 tokenizer_kwargs=None,
                 model_kwargs=None,
                 generation_kwargs=None,
                 on_progress=None,
                 response_re=None,
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
            response_re=response_re,
        )
        self.model_id = model_id
        # load_path is the local filesystem path for from_pretrained; falls back to model_id.
        self.load_path = load_path or model_id
        self.device_param = device
        self.processor_kwargs = processor_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.generation_kwargs = generation_kwargs or {}
        self._on_progress = on_progress
        self._setup_model()

    def _select_device(self):
        cuda_available = torch.cuda.is_available()
        mps_available = not cuda_available and torch.backends.mps.is_available()
        return cuda_available, "cuda" if cuda_available else ("mps" if mps_available else "cpu")

    def _select_quant(self, cuda_available):
        return BitsAndBytesConfig(load_in_4bit=True) if (cuda_available and importlib.util.find_spec('bitsandbytes')) else None

    def _select_attn(self):
        if importlib.util.find_spec('flash_attn'):
            if _flash_attn2_compatible(self.load_path):
                return 'flash_attention_2'
            log.info("flash_attn installed but model is not flash_attn 2 compatible; using sdpa")
        return 'sdpa'

    @contextlib.contextmanager
    def _tqdm_patched(self):
        if not self._on_progress:
            yield
            return
        import importlib as _il
        _WeightTqdm = self._make_weight_tqdm()
        patches = []
        for mod_name, attr in [('tqdm', 'tqdm'), ('tqdm.auto', 'tqdm'),
                                ('transformers.modeling_utils', 'tqdm')]:
            try:
                mod = _il.import_module(mod_name)
                if hasattr(mod, attr):
                    patches.append((mod, attr, getattr(mod, attr)))
                    setattr(mod, attr, _WeightTqdm)
            except ImportError:
                pass
        try:
            yield
        finally:
            for mod, attr, orig in patches:
                setattr(mod, attr, orig)

    def _maybe_enable_xformers(self, model, attn_implementation):
        # Skip when flash_attention_2 is in use — it outperforms xformers and
        # enable_xformers_memory_efficient_attention() would silently override it.
        if attn_implementation != "flash_attention_2" and importlib.util.find_spec('xformers'):
            try:
                model.enable_xformers_memory_efficient_attention()
                self._notify("xformers memory-efficient attention enabled")
            except Exception as e:
                log.debug("xformers enable skipped: %s", e)

    def _setup_model(self):
        cuda_available, device = self._select_device()
        quantization_config = self._select_quant(cuda_available)
        attn_implementation = self._select_attn()

        self._notify(f"Loading {self._display_name()} (device={device}, quant={quantization_config is not None}, attn={attn_implementation})")

        with self._tqdm_patched():
            self.model = self._create_model(quantization_config, attn_implementation, device)

        self._notify("Model loaded")
        self._maybe_enable_xformers(self.model, attn_implementation)

        try:
            from huggingface_hub import snapshot_download
            log.info("Model directory: %s", snapshot_download(self.model_id, local_files_only=True))
        except Exception:
            pass

        if device != "cuda":
            device = torch.device(self.device_param or device)
            self._notify(f"Moving model to {device}")
            try:
                self.model = self.model.to(device)
            except TypeError:
                self.model = self.model.to(device, dtype=torch.float16)

        if self.uses_processor:
            self._notify("Loading processor")
            processor_kwargs = {"trust_remote_code": True} | self.processor_extra_kwargs | self.processor_kwargs
            self.processor = AutoProcessor.from_pretrained(self.load_path, **processor_kwargs)
        try:
            self.tokenizer = self.processor.tokenizer
        except AttributeError:
            tokenizer_kwargs = {"trust_remote_code": True} | self.tokenizer_extra_kwargs | self.tokenizer_kwargs
            self.tokenizer = AutoTokenizer.from_pretrained(self.load_path, **tokenizer_kwargs)

        self.generation_args = self.default_generation_args | self.generation_kwargs

        model_params = next(self.model.parameters())
        self.device = model_params.device
        self.dtype = model_params.dtype
        self._notify(f"{self._display_name()} ready (device={self.device}, dtype={self.dtype})")

    def _create_model(self, quantization_config, attn_implementation, device):
        model_kwargs = {
            "device_map": "cuda" if device == "cuda" and not self.device_param else None,
            "trust_remote_code": True,
            "quantization_config": quantization_config,
            "torch_dtype": "auto",
            "_attn_implementation": attn_implementation,
        } | self.model_kwargs
        log.info("model_kwargs: %s", model_kwargs)
        return AutoModelForCausalLM.from_pretrained(self.load_path, **model_kwargs)

    @abstractmethod
    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        pass
