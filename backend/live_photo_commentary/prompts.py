import yaml
from pathlib import Path

from .describer import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_PROMPT,
    DEFAULT_FIRST_PROMPT,
    DEFAULT_HISTORY_PROMPT,
    DEFAULT_COMPACT_PROMPT,
    TAGS_PLACEHOLDER,
)

FIELDS = ["system_prompt", "prompt", "first_prompt", "history_prompt", "compact_prompt"]
DEFAULT_NAME = "default"

_dir: Path = Path("prompts")


def set_dir(path: Path) -> None:
    global _dir
    _dir = path
    path.mkdir(parents=True, exist_ok=True)


def defaults() -> dict:
    return {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "prompt": DEFAULT_PROMPT,
        "first_prompt": DEFAULT_FIRST_PROMPT,
        "history_prompt": DEFAULT_HISTORY_PROMPT,
        "compact_prompt": DEFAULT_COMPACT_PROMPT,
    }


def list_names() -> list[str]:
    names = sorted(p.stem for p in _dir.glob("*.yaml"))
    return [DEFAULT_NAME] + names


def load(name: str) -> dict:
    if name == DEFAULT_NAME:
        return defaults()
    path = _dir / f"{name}.yaml"
    if not path.exists():
        return defaults()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base = defaults()
    return {k: data.get(k, base[k]) for k in FIELDS}


def save(name: str, fields: dict) -> None:
    if name == DEFAULT_NAME:
        raise ValueError("Cannot overwrite the default promptset")
    path = _dir / f"{name}.yaml"
    data = {k: fields.get(k, "") for k in FIELDS}
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, width=10000)


def substitute_tags(fields: dict, tag_names: list[str]) -> dict:
    """Replace TAGS_PLACEHOLDER in each field with the available emotion tags,
    formatted as {tag} marks. "neutral" is always included."""
    names = list(dict.fromkeys([*tag_names, "neutral"]))
    tag_list = ", ".join(f"{{{name}}}" for name in names)
    return {
        k: v.replace(TAGS_PLACEHOLDER, tag_list) if isinstance(v, str) else v
        for k, v in fields.items()
    }


def delete(name: str) -> None:
    if name == DEFAULT_NAME:
        raise ValueError("Cannot delete the default promptset")
    path = _dir / f"{name}.yaml"
    if path.exists():
        path.unlink()
