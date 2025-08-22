import importlib.util

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



class LocalDescriber(Describer):
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

        quantization_config = BitsAndBytesConfig(load_in_4bit=True) if importlib.util.find_spec('bitsandbytes') else None

        attn_implementation = 'flash_attention_2' if importlib.util.find_spec('flash_attn') else 'eager'
        cuda_available = torch.cuda.is_available()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda" if cuda_available else None, 
            trust_remote_code=True, 
            quantization_config=quantization_config,
            torch_dtype="auto", 
            _attn_implementation=attn_implementation
        )

        if not cuda_available:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            # Fallback dtype
            try:
                self.model = self.model.to(device)
            except TypeError:
                self.model = self.model.to(device, dtype=torch.float16)

        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        self.processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            num_crops=4
        )
        try:
            self.tokenizer = self.processor.tokenizer
        except KeyError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )

        self.generation_args = { 
            "max_new_tokens": 200, 
            "temperature": 0.2, 
            "do_sample": True, 
        }

        self.device = next(self.model.parameters()).device

    def prompt_model(self, user_prompt, images=None, placeholder=None) -> str | None:
        if images is not None:
            user_prompt = placeholder +  user_prompt.replace("<|image_1|>","first image").replace("<|image_2|>","second image")

        messages = [
            { "role": "system", "content": self.system_prompt },
            { "role": "user", "content": user_prompt },
        ]

        prompt = self.tokenizer.apply_chat_template(
          messages, 
          tokenize=False, 
          add_generation_prompt=True
        )

        inputs = self.processor(prompt, images, return_tensors="pt").to(self.device)

        # Avoid Phi bug on MPS
        if 'image_sizes' in inputs:
            inputs['image_sizes'] = inputs['image_sizes'].tolist()

        generate_ids = self.model.generate(**inputs, 
          eos_token_id=self.tokenizer.eos_token_id, 
          **self.generation_args
        )

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response_text = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        return response_text
