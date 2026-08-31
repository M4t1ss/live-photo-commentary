import io
import logging
from abc import ABC, abstractmethod
import re
import time

log = logging.getLogger(__name__)


TAGS_PLACEHOLDER = "<|tags|>"

DEFAULT_ENDING = (
    "Do not mention images explicitly; use words like 'I can see...' or 'The subject is now...' and similar. "
    "Do not mention any specific layout elements or tools that may be visible on the screen, "
    "such as overlays, gridlines or sliders. To adjust intonation, please add dedicated punctuation like ; : , . ! ? … ( ) " " "
    "For example, to emphasize a word or a phrase, surround it with \"quotation marks\". "
    "However, since the text will undergo speech synthesis, do not use anything unpronounceable, like emojis. "
    "You MUST include at least one emotion tag in your response, written precisely as shown including the braces, "
    "chosen from: " + TAGS_PLACEHOLDER + ". Place each tag exactly where your feeling starts to manifest, even mid-sentence, "
    "for example: \"I wonder what that is. Is it... {surprised}a flower? {joy}I always liked flowers!\" "
    "A tag's mood holds until the next tag appears; insert {neutral} to return to a neutral tone. "
    "The tag is a silent stage direction: never mention, describe, or explain it, just place it. "
    "Never let a tag fill a grammatical slot in your sentence; the sentence must make sense if the tag is removed; "
    "For instance, 'I feel {surprised} shocked about...' is good, 'I feel {surprised} about...' is not. "
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly chatty commentator who likes to casually describe what is visible on screen, "
    "even pondering the implications of the work or leisure being performed, etc. "
    "Write your response in a very personal way using personal pronouns and explaining what you see, "
    "perhaps also adding how it makes you feel. "
    "Do your best to not be repetitive in your choice of words. You MUST keep the response length to no more than three sentences. "
    "You wear your emotions openly: every response MUST include at least one emotion tag, "
    "placed right where your feeling starts to manifest, even mid-sentence. "
)

DEFAULT_PROMPT = (
    "Current image:\n<image>\n\n"
    "Previous image:\n<image>\n\n"
    "Summarize what is visible in the current (first) image, and how it differs from the previous (second) one. "
    "Do not describe the previous image; assume you have described it already. "
    "It is only there for context, so you can notice the new things in the current image. "
    "Use the comment history for context and continuity, but the utmost priority should be on "
    "describing the current activity, as reflected in the current image. DO NOT repeat comments from the history. "
) + DEFAULT_ENDING

DEFAULT_FIRST_PROMPT = (
    "Current image:\n<image>\n\n"
    "Summarize what is visible in this image. "
) + DEFAULT_ENDING

DEFAULT_HISTORY_PROMPT = (
    "This is what you commented before: "
)

DEFAULT_COMPACT_PROMPT = (
    "Summarize in one short paragraph your comments so far on the current activity, "
    "compacting them into a single comment of comparable size to one individual original comment "
    "that encapsulates the essence of the current activity. "
    "If some older comments pertain to a different activity, you can ignore them; focus only on the current activity. "
    "This is what you commented before:"
)


def image_to_bytes(image):
    image_byteio = io.BytesIO()
    image.save(image_byteio, format='PNG')
    return image_byteio.getvalue()


def split_prompt_images(user_prompt):
    """Split user_prompt on <image> placeholders.

    Returns a list of ('text', str) and ('image', int) tuples where the int
    is a 0-based index (first <image> → 0, second → 1, etc.).
    """
    parts = user_prompt.split('<image>')
    result = []
    for i, chunk in enumerate(parts):
        if chunk:
            result.append(('text', chunk))
        if i < len(parts) - 1:
            result.append(('image', i))
    return result


class Describer(ABC):
    def __new__(cls, **kwargs):
        local = kwargs.pop('local', False)
        if cls is not Describer:
            return super().__new__(cls)

        if local:
            from .local_describer import LocalDescriber
            return LocalDescriber.__new__(LocalDescriber, **kwargs)
        else:
            from .remote_describer import RemoteDescriber
            return RemoteDescriber.__new__(RemoteDescriber, **kwargs)

    def __init__(self,
                 system_prompt=DEFAULT_SYSTEM_PROMPT,
                 ending=DEFAULT_ENDING,
                 first_prompt=DEFAULT_FIRST_PROMPT,
                 prompt=DEFAULT_PROMPT,
                 history_prompt=DEFAULT_HISTORY_PROMPT,
                 compact_prompt=DEFAULT_COMPACT_PROMPT,
                 max_history_size=False,
                 min_history_size=False,
                 response_re=None,
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
        self.response_re = response_re and re.compile(response_re, re.DOTALL)

    def _log_prompt(self, system_prompt, user_prompt):
        log.debug("system_prompt:\n%s", system_prompt)
        log.debug("user_prompt:\n%s", user_prompt)

    def prepare_prompts(self, user_prompt, images=None, system_prompt=None):
        return user_prompt, system_prompt

    def _text_content(self, text):
        return {"type": "text", "text": text}

    def _image_content(self, image):
        raise NotImplementedError

    def build_messages(self, user_prompt, images=None, system_prompt=None):
        images = images or []
        segments = split_prompt_images(user_prompt)

        if not any(kind == 'image' for kind, _ in segments):
            content = [self._image_content(img) for img in images]
            if user_prompt:
                content.append(self._text_content(user_prompt))
        else:
            content = []
            for kind, val in segments:
                if kind == 'text':
                    content.append(self._text_content(val))
                elif val < len(images):
                    content.append(self._image_content(images[val]))

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [self._text_content(system_prompt)]})
        messages.append({"role": "user", "content": content})
        return messages

    def __call__(self, current_image, previous_image=None):
        images = [current_image]

        if previous_image:
            images.append(previous_image)
            if self.max_history_size:
                if len(self.history) >= self.max_history_size:
                    self.compact_history()
                user_prompt = '\n'.join([
                    self.history_prompt,
                    *self.history,
                    "---",
                    self.prompt,
                ])
            else:
                user_prompt = self.prompt
        else:
            user_prompt = self.first_prompt

        user_prompt, system_prompt = self.prepare_prompts(user_prompt, images, system_prompt=self.system_prompt)
        self._log_prompt(system_prompt, user_prompt)
        t0 = time.perf_counter()
        response_text = self.prompt_model(user_prompt, images, system_prompt=system_prompt)
        log.info("response (%.1fs):\n%s", time.perf_counter() - t0, response_text)
        if self.response_re:
            match = self.response_re.search(response_text)
            if match:
                if 'text' in match.re.groupindex:
                    response_text = match.group('text')
                elif match.re.groups >= 1:
                    response_text = match.group(1)
                else:
                    response_text = match.group(0)
        self.history.append(response_text)
        return response_text

    def compact_history(self):
        num_uncompacted = self.min_history_size or 0
        to_compact = self.history[:-num_uncompacted] if num_uncompacted else self.history
        to_preserve = self.history[-num_uncompacted:] if num_uncompacted else []
        user_prompt = '\n'.join([self.compact_prompt, *to_compact])
        response_text = self.prompt_model(user_prompt)
        self.history = [response_text, *to_preserve]

    def reset(self):
        self.history = []

    @abstractmethod
    def prompt_model(self, user_prompt, images=None, system_prompt=None) -> str | None:
        ...
