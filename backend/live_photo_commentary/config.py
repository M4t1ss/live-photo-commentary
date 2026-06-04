from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vlm_provider: str = "gemini"
    vlm_model: str | None = None
    tts_voice: str = "af_heart"
    post_speech_delay: float = 2.0
    difference_threshold: float = 0.0
    difference_measure: str = "mse"
    max_history_size: int = 0
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    elevenlabs_api_key: str | None = None


_API_KEY_FIELDS = frozenset({"gemini_api_key", "openai_api_key", "elevenlabs_api_key"})

_settings = Settings()


def get() -> Settings:
    return _settings


def apply(updates: dict) -> Settings:
    global _settings
    known = {k: v for k, v in updates.items() if k in Settings.model_fields}
    _settings = _settings.model_copy(update=known)
    return _settings


def as_dict(masked: bool = True) -> dict:
    d = _settings.model_dump()
    if masked:
        for field in _API_KEY_FIELDS:
            if d[field] is not None:
                d[field] = "***"
    return d


def persist(path: Path) -> None:
    lines = [
        f"{field.upper()}={value}\n"
        for field, value in _settings.model_dump().items()
        if value is not None
    ]
    path.write_text("".join(lines), encoding="utf-8")
