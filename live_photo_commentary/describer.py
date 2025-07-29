import os
import io
import importlib.util

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.utils.quantization_config import BitsAndBytesConfig
from google import genai
from google.genai import types as genai_types



DEFAULT_ENDING = (
    "Do not at all mention any specific layout elements or tools that may be visible on the screen, "
    "such as overlays, gridlines or sliders. To adjust intonation, please add dedicated punctuation like ; : , . ! ? … ( ) “ ” "
    "For example, to emphasize a word or a phrase, surround it with \"quotation marks\". "
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly chatty commentator who likes to casually describe work done by a computer user " 
    "in various details, even by pondering the implications on work, or leisure, being performed, etc. Write your " 
    "response in a very personal way using personal pronouns and explaining what you see, perhaps also adding how it makes you feel. " 
    "Do your best to not be repetitive in your choice of words and keep the response length down to a few sentences. You MUST NOT mention "
    "any specific layout elements or tools that may be visible on the screen, such as gridlines or sliders. "
) # + DEFAULT_ENDING

DEFAULT_PROMPT = (
    "Summarize what is visible in the current screenshot, <|image_1|>. " 
    "How is it different from the previous screenshot, <|image_2|>? There may be some subtle differences as well. Do not describe the previous screenshot; assume you have described it already. It is only there for context, so you can notice the new things in the current screenshot. Do not mention screenshots explicitly; use words like 'I can see...' or 'The user is now...' and similar."
) + DEFAULT_ENDING

DEFAULT_FIRST_PROMPT = (
    "Summarize what is visible in this image: <|image_1|> "
) + DEFAULT_ENDING

def image_to_bytes(image):
    image_byteio = io.BytesIO()
    image.save(image_byteio, format='PNG')
    image_bytes = image_byteio.getvalue()
    return image_bytes


class Describer:
    def __init__(self,
             system_prompt=DEFAULT_SYSTEM_PROMPT,
             ending=DEFAULT_ENDING,
             first_prompt=DEFAULT_FIRST_PROMPT,
             prompt=DEFAULT_PROMPT,
             local=False, gemini_api_key=None
    ):
        self.ending = ending
        self.system_prompt = system_prompt
        self.first_prompt = first_prompt
        self.prompt = prompt

        if local:
            model_id = "microsoft/Phi-3.5-vision-instruct" 
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            
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
                self.model = self.model.to(device)

            # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
            self.processor = AutoProcessor.from_pretrained(model_id, 
                trust_remote_code=True, 
                num_crops=4
            )

            self.generation_args = { 
                "max_new_tokens": 200, 
                "temperature": 0.2, 
                "do_sample": True, 
            }

            self.device = next(self.model.parameters()).device
        else:
            self.model = self.processor = None
            if not gemini_api_key:
                gemini_api_key = os.environ["GEMINI_API_KEY"]
            self.client = genai.Client(api_key=gemini_api_key)


    def __call__(self, current_image, previous_image=None):
        images = []

        images.append(image_to_bytes(current_image))
        if previous_image:
            images.append(image_to_bytes(previous_image))
            user_prompt = self.prompt
        else:
            user_prompt = self.first_prompt

        if self.model and self.processor:
        
            messages = [
                { "role": "system", "content": self.system_prompt },
                { "role": "user", "content": user_prompt },
            ]
        
            prompt = self.processor.tokenizer.apply_chat_template(
              messages, 
              tokenize=False, 
              add_generation_prompt=True
            )
            
            inputs = self.processor(prompt, images, return_tensors="pt").to(self.device)
            
            generate_ids = self.model.generate(**inputs, 
              eos_token_id=self.processor.tokenizer.eos_token_id, 
              **self.generation_args
            )
            
            # remove input tokens 
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(
                generate_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return response
        else:
            # Create the prompt with text and multiple images
            contents: genai_types.ContentListUnion = [
                genai_types.Part.from_bytes(data=image, mime_type='image/png') for image in images
            ]
            contents.insert(0, user_prompt)

            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                config=genai_types.GenerateContentConfig(system_instruction = self.system_prompt),
                contents=contents,
            )

            return response.text

if __name__ == '__main__':
    from PIL import ImageGrab
    screenshot = ImageGrab.grab()
    describer = Describer()
    description = describer(screenshot)
    print(description)
