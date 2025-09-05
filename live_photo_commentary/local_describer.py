import importlib.util
from abc import abstractmethod
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
    """Convert PIL image to data URI."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    image_bytes = buffer.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:image/png;base64,{image_base64}"


class LocalDescriber(Describer):
    """Abstract base class for local vision models with factory pattern."""

    uses_processor = True
    
    def __new__(cls, model_id="microsoft/Phi-3.5-vision-instruct", **kwargs):
        """Factory method that returns the appropriate subclass based on model_id."""
        if cls is not LocalDescriber:
            # Direct instantiation of subclass
            return super().__new__(cls)
        
        # Factory logic for LocalDescriber instantiation with lazy imports
        if "FastVLM" in model_id:
            from .local_describers.fastvlm import FastVLMLocalDescriber
            instance = super(LocalDescriber, FastVLMLocalDescriber).__new__(FastVLMLocalDescriber)
        elif "Qwen" in model_id or "gemma" in model_id.lower():
            from .local_describers.gemma3 import Gemma3LocalDescriber
            instance = super(LocalDescriber, Gemma3LocalDescriber).__new__(Gemma3LocalDescriber)
        elif "Phi-4" in model_id:
            from .local_describers.phi4mm import Phi4MMLocalDescriber
            instance = super(LocalDescriber, Phi4MMLocalDescriber).__new__(Phi4MMLocalDescriber)
        elif "Phi-3" in model_id:
            from .local_describers.phi3v import Phi3VLocalDescriber
            instance = super(LocalDescriber, Phi3VLocalDescriber).__new__(Phi3VLocalDescriber)
        else:
            raise ValueError(
                f"Unsupported model: {model_id}. "
                f"Supported models: FastVLM, Phi-3.x, Phi-4.x, Gemma, Qwen"
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
                 model_id="microsoft/Phi-3.5-vision-instruct",
                 device=None,
                 processor_kwargs=None,
                 tokenizer_kwargs=None,
                 model_kwargs=None,
                 generation_kwargs=None,
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
        self._setup_model()
        
    def _setup_model(self):
        """Common model setup logic."""
        quantization_config = BitsAndBytesConfig(load_in_4bit=True) if "Phi-4" not in self.model_id and importlib.util.find_spec('bitsandbytes') else None
        attn_implementation = 'flash_attention_2' if importlib.util.find_spec('flash_attn') else 'eager'
        cuda_available = torch.cuda.is_available()
        
        self.model = self._create_model(quantization_config, attn_implementation, cuda_available)
        
        if not cuda_available:
            device = torch.device(self.device_param or ("mps" if torch.backends.mps.is_available() else "cpu"))
            try:
                self.model = self.model.to(device)
            except TypeError:
                self.model = self.model.to(device, dtype=torch.float16)
        
        if self.uses_processor:
            processor_kwargs = {
                "trust_remote_code": True,
                "num_crops": 4
            } | self.processor_kwargs
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                **processor_kwargs
            )
        try:
            self.tokenizer = self.processor.tokenizer
        except AttributeError:
            tokenizer_kwargs = {
                "trust_remote_code": True,
            } | self.tokenizer_kwargs
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                **tokenizer_kwargs
            )
        
        self.generation_args = {
            "max_new_tokens": 200,
            "temperature": 0.2,
            "do_sample": True,
        } | self.generation_kwargs
        
        model_params = next(self.model.parameters())
        self.device = model_params.device
        self.dtype = model_params.dtype
    
    def _create_model(self, quantization_config, attn_implementation, cuda_available):
        """Create the model. Subclasses can override this for model-specific logic."""
        model_kwargs = {
            "device_map": "cuda" if cuda_available and not self.device_param else None,
            "trust_remote_code": True,
            "quantization_config": quantization_config,
            "torch_dtype": "auto",
            "_attn_implementation": attn_implementation
        } | self.model_kwargs
        return AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )
    
    @abstractmethod
    def prompt_model(self, user_prompt, images=None) -> str | None:
        """Abstract method to be implemented by subclasses."""
        pass
