import io
from abc import ABC, abstractmethod



DEFAULT_ENDING = (
    "Do not at all mention any specific layout elements or tools that may be visible on the screen, "
    "such as overlays, gridlines or sliders. To adjust intonation, please add dedicated punctuation like ; : , . ! ? … ( ) “ ” "
    "For example, to emphasize a word or a phrase, surround it with \"quotation marks\". "
    "However, since the text will undergo speech synthesis, do not use anything unpronounceable, like emojis."
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

DEFAULT_HISTORY_PROMPT = (
    "This is what you (the assistant) commented before, for context:"
)

DEFAULT_COMPACT_PROMPT = (
    "Summarize in one short paragraph your (the assistant's) comments so far on the current activity; "
    "i.e. compact it into a single comment, that encapsulates the essence of the current activity's past. "
    "If some older comments pertain to a different activity, you can ignore them; focus only on the current activity. "
    "This is what you (the assistant) commented before:"
)


def image_to_bytes(image):
    image_byteio = io.BytesIO()
    image.save(image_byteio, format='PNG')
    image_bytes = image_byteio.getvalue()
    return image_bytes


class Describer(ABC):
    def __init__(self,
             system_prompt=DEFAULT_SYSTEM_PROMPT,
             ending=DEFAULT_ENDING,
             first_prompt=DEFAULT_FIRST_PROMPT,
             prompt=DEFAULT_PROMPT,
             history_prompt=DEFAULT_HISTORY_PROMPT,
             compact_prompt=DEFAULT_COMPACT_PROMPT,
             max_history_size=False,
    ):
        self.ending = ending
        self.system_prompt = system_prompt
        self.first_prompt = first_prompt
        self.prompt = prompt
        self.history_prompt = history_prompt
        self.compact_prompt = compact_prompt
        self.max_history_size = max_history_size

        self.history = []


    def __call__(self, current_image, previous_image=None):
        images = []

        images.append(current_image)
        if previous_image:
            images.append(previous_image)
            if self.max_history_size:
                if len(self.history) >= self.max_history_size:
                    self.compact_history()
                user_prompt = '\n'.join([
                    self.history_prompt,
                    *self.history,
                    "---",
                    self.prompt
                ])
            else:
                user_prompt = self.prompt
        else:
            user_prompt = self.first_prompt

        response_text = self.prompt_model(user_prompt, images)
        self.history.append(response_text)
        return response_text


    def compact_history(self):
        user_prompt = '\n'.join([
            self.compact_prompt,
            *self.history[:-1],
        ])
        response_text = self.prompt_model(user_prompt)
        self.history = [
            response_text,
            self.history[-1],
        ]


    @abstractmethod
    def prompt_model(self, user_prompt, images=None) -> str | None:
        ...
