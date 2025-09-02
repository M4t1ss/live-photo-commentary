"""FastVLM LocalDescriber implementation."""

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..local_describer import LocalDescriber


class FastVLMLocalDescriber(LocalDescriber):
    """LocalDescriber for Apple's FastVLM models."""
    
    IMAGE_TOKEN_INDEX = -200  # Special token index for FastVLM
    
    def _create_model(self, quantization_config, attn_implementation, cuda_available):
        """Create FastVLM model with specific requirements."""
        return AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cuda" if cuda_available and not self.device_param else "auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if cuda_available else torch.float32,
        )
    
    def _setup_model(self):
        """Override setup to use AutoTokenizer directly for FastVLM."""
        quantization_config = None
        if hasattr(torch.backends, 'cuda') and torch.cuda.is_available():
            try:
                import bitsandbytes
                from transformers.utils.quantization_config import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            except ImportError:
                pass
        
        cuda_available = torch.cuda.is_available()
        
        self.model = self._create_model(quantization_config, None, cuda_available)
        
        if not cuda_available:
            device = torch.device(self.device_param or ("mps" if torch.backends.mps.is_available() else "cpu"))
            try:
                self.model = self.model.to(device)
            except TypeError:
                self.model = self.model.to(device, dtype=torch.float16)
        
        # FastVLM uses AutoTokenizer directly
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_fast=False
        )
        
        # No separate processor needed for FastVLM
        self.processor = None
        
        self.generation_args = {
            "max_new_tokens": 200,
            "temperature": 0.2,
            "do_sample": True,
        }
        
        self.device = next(self.model.parameters()).device
    
    def prompt_model(self, user_prompt, images=None) -> str | None:
        if not images:
            images = []
        
        user_prompt = re.sub(r'<\|image_\d+\|>', '<image>', user_prompt)
        
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        parts = rendered.split("<image>")
        input_ids_parts = []
        
        for i in range(len(images)):
            # text tokens
            if parts[i]:
                text_ids = self.tokenizer(parts[i], return_tensors="pt", add_special_tokens=False).input_ids
                input_ids_parts.append(text_ids)

            # image token
            img_tok = torch.tensor([[self.IMAGE_TOKEN_INDEX]], dtype=torch.long)
            input_ids_parts.append(img_tok)
            
        if parts[-1]:
            text_ids = self.tokenizer(parts[-1], return_tensors="pt", add_special_tokens=False).input_ids
            input_ids_parts.append(text_ids)
        
        
        input_ids = torch.cat(input_ids_parts, dim=1).to(self.device)
        
        # Process images using vision tower image processor
        images = [image.convert('RGB') for image in images]
        if images:
            images_tensor = self.model.get_vision_tower().image_processor(images=images, return_tensors="pt")["pixel_values"]
            images_tensor = images_tensor.to(self.device, dtype=self.model.dtype)
        else:
            images_tensor = None
        
        attention_mask = torch.ones_like(input_ids, device=self.device)
        
        input_len = input_ids.shape[-1]
        
        with torch.inference_mode():
            generate_kwargs = {
                "inputs": input_ids,
                "images": images_tensor,
                "attention_mask": attention_mask,
                "eos_token_id": self.tokenizer.eos_token_id,
                **self.generation_args
            }
            
            generate_ids = self.model.generate(**generate_kwargs)
        
        # Remove input tokens and decode
        # generate_ids = generate_ids[:, input_len:]
        response_text = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        return response_text
