from pathlib import Path
import re
import yaml

DEFAULT_NAME = "default"

_model_dir: Path = Path("../models")
_user_dir: Path = Path("user_models")


def set_dirs(model_dir: Path, user_dir: Path) -> None:
    global _model_dir, _user_dir
    _model_dir = model_dir
    _user_dir = user_dir
    _user_dir.mkdir(parents=True, exist_ok=True)


def get_user_dir() -> Path:
    return _user_dir


def list_names() -> list[str]:
    bundled = sorted(
        p.stem for p in _model_dir.glob("*.glb")
        if p.stem != "model"
    ) if _model_dir.exists() else []
    user = sorted(
        f"user/{p.stem}" for p in _user_dir.glob("*.glb")
    ) if _user_dir.exists() else []
    return [DEFAULT_NAME] + bundled + user


def resolve(name: str) -> tuple[Path, Path]:
    """Return (glb_path, yaml_path). yaml_path always falls back to model_dir/model.yaml."""
    fallback_yaml = _model_dir / "model.yaml"
    if not name or name == DEFAULT_NAME:
        return _model_dir / "model.glb", fallback_yaml
    if name.startswith("user/"):
        stem = name[5:]
        yaml_path = _user_dir / f"{stem}.yaml"
        return _user_dir / f"{stem}.glb", yaml_path if yaml_path.exists() else fallback_yaml
    yaml_path = _model_dir / f"{name}.yaml"
    return _model_dir / f"{name}.glb", yaml_path if yaml_path.exists() else fallback_yaml


def save_user_model(filename: str, data: bytes) -> str:
    """Store a VRM/GLB file in the user models directory. Returns the model name."""
    _user_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    stem = re.sub(r"[^\w\-]", "_", stem, flags=re.ASCII)
    (_user_dir / f"{stem}.glb").write_bytes(data)
    return f"user/{stem}"


def load_tag_names(name: str) -> list[str]:
    _, yaml_path = resolve(name)
    if not yaml_path.exists():
        return []
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return list(raw.get("tags", {}).keys())
