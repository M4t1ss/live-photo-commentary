import re

from semantic_text_splitter import TextSplitter

_TAG_RE = re.compile(r'\{[^}]*\}')


def _visible_len(text: str) -> int:
    if text.count('{') != text.count('}'):
        return 2 ** 31  # never split inside a tag
    return len(_TAG_RE.sub('', text))


def insert_subtitle_tags(text: str, max_chars: int = 60, tag: str = "sub") -> str:
    """
    Insert {tag} markers between subtitle segments so each segment's visible
    character count (excluding any {tags}) stays within max_chars.

    Uses semantic_text_splitter, which splits at sentence boundaries first,
    then word boundaries, so breaks prefer natural sentence ends.
    """
    splitter = TextSplitter.from_callback(_visible_len, capacity=max_chars)
    chunks = list(splitter.chunks(text))
    return ("{" + tag + "}").join(chunks)
