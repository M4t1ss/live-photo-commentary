import regex

from semantic_text_splitter import TextSplitter

_TAG_RE = regex.compile(r'\{[^}]*\}')
_WHITESPACE_PLUS_RE = regex.compile(r' {2,}')
_CONTENT_RE = regex.compile(r'[\p{L}\p{N}]')


def has_content(text: str) -> bool:
    """Return True if text contains at least one letter or digit outside of tags."""
    return bool(_CONTENT_RE.search(_TAG_RE.sub('', text)))


def _sub_splitter_callback(text: str) -> int:
    # subtitle splitting is based on text size vs subtitle capacity
    # deny splitting within tags
    if text.count('{') != text.count('}'):
        return 2 ** 31
    # ignore tags
    braceless = _TAG_RE.sub('', text).strip()
    braceless = _WHITESPACE_PLUS_RE.sub(' ', braceless)
    # deny content-free chunks (only punctuation/whitespace outside tags)
    if not _CONTENT_RE.search(braceless):
        return 2 ** 31
    return len(braceless)


def _shift_whitespace(chunks):
    if not chunks:
        return []

    result = list(chunks)

    for i in range(1, len(result)):
        item = result[i]
        stripped = item.lstrip()
        whitespace_len = len(item) - len(stripped)
        if whitespace_len:
            whitespace = item[:whitespace_len]
            result[i - 1] += whitespace
            result[i] = stripped

    return result

def insert_subtitle_tags(text: str, max_chars: int = 60, tag: str = "sub") -> str:
    """
    Insert {tag} markers between subtitle segments so each segment's visible
    character count (excluding any {tags}) stays within max_chars.

    Uses semantic_text_splitter, which splits at sentence boundaries first,
    then word boundaries, so breaks prefer natural sentence ends.
    """
    splitter = TextSplitter.from_callback(_sub_splitter_callback, capacity=max_chars, trim=False)
    chunks = list(splitter.chunks(text))
    chunks = _shift_whitespace(chunks)
    return ("{" + tag + "}").join(chunks)
