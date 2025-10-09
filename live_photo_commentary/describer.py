import io
from abc import ABC, abstractmethod



DEFAULT_ENDING = (
    "Do not at all mention any specific layout elements or tools that may be visible on the screen, "
    "such as overlays, gridlines or sliders. To adjust intonation, please add dedicated punctuation like ; : , . ! ? … ( ) “ ” "
    "For example, to emphasize a word or a phrase, surround it with \"quotation marks\". "
    "However, since the text will undergo speech synthesis, do not use anything unpronounceable, like emojis."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly chatty commentator who likes to casually describe work done by a photographer " 
    "in various details, even by pondering the implications on work, or leisure, being performed, etc. "
    "Write your response in a very personal way using personal pronouns and explaining what you see, "
    "perhaps also adding how it makes you feel. " 
    "Do your best to not be repetitive in your choice of words. You MUST keep the response length to no more than three sentences. "
    "You MUST NOT mention any specific layout elements or tools that may be visible on the screen, such as gridlines or sliders. "
) # + DEFAULT_ENDING

DEFAULT_PROMPT = (
    "Summarize what is visible in the current photo, <|image_1|>. " 
    "How is it different from the previous photo, <|image_2|>? "
    "There may be some subtle differences as well. "
    "Do not describe the previous photo; assume you have described it already. "
    "It is only there for context, so you can notice the new things in the current photo. "
    "Do not mention photos explicitly; use words like 'I can see...' or 'The photographer is now...' and similar. "
    "Use the comment history for context and continuity, but the utmost priority should be on "
    "describing the current activity, as reflected in the current photo. DO NOT repeat comments from the history. "
) + DEFAULT_ENDING

DEFAULT_FIRST_PROMPT = (
    "Summarize what is visible in this image: <|image_1|> "
) + DEFAULT_ENDING

DEFAULT_HISTORY_PROMPT = (
    "This is what you (the assistant) commented before: "
)

DEFAULT_COMPACT_PROMPT = (
    "Summarize in one short paragraph your (the assistant's) comments so far on the current activity; "
    "i.e. compact it into a single comment of comparable size to one individual original comment, "
    "that encapsulates the essence of the current activity's past. "
    "If some older comments pertain to a different activity, you can ignore them; focus only on the current activity. "
    "This is what you (the assistant) commented before:"
)


def image_to_bytes(image):
    image_byteio = io.BytesIO()
    image.save(image_byteio, format='PNG')
    image_bytes = image_byteio.getvalue()
    return image_bytes


class Describer(ABC):
    def __new__(cls, **kwargs):
        """Factory method that returns the appropriate subclass based on local parameter."""
        local = kwargs.pop('local', False)
        if cls is not Describer:
            # Direct instantiation of subclass
            return super().__new__(cls)
        
        if local:
            from .local_describer import LocalDescriber
            return LocalDescriber(**kwargs)
        else:
            from .remote_describer import RemoteDescriber
            return RemoteDescriber(**kwargs)

    def __init__(self,
             system_prompt=DEFAULT_SYSTEM_PROMPT,
             ending=DEFAULT_ENDING,
             first_prompt=DEFAULT_FIRST_PROMPT,
             prompt=DEFAULT_PROMPT,
             history_prompt=DEFAULT_HISTORY_PROMPT,
             compact_prompt=DEFAULT_COMPACT_PROMPT,
             max_history_size=False,
             min_history_size=False,
    ):
        self.ending = ending
        self.system_prompt = system_prompt
        self.first_prompt = first_prompt
        self.prompt = prompt
        self.history_prompt = history_prompt
        self.compact_prompt = compact_prompt
        self.max_history_size = max_history_size
        self.min_history_size = min_history_size

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

        response_text = self.prompt_model(user_prompt, images, system_prompt=self.system_prompt)
        self.history.append(response_text)
        return response_text


    def compact_history(self):
        num_uncompacted = self.min_history_size or 0
        uncompacted = len(self.history) - num_uncompacted
        to_compact = self.history[:uncompacted]
        to_preserve = self.history[uncompacted:]
        user_prompt = '\n'.join([
            self.compact_prompt,
            *to_compact,
        ])
        response_text = self.prompt_model(user_prompt)
        self.history = [
            response_text,
            *to_preserve,
        ]

    def reset(self):
        """Reset the describer to its initial state."""
        self.history = []

    @abstractmethod
    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        ...
