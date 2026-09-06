import hashlib
import json
import logging
from pathlib import Path

from .baseline import SUPPORTED_EXTS

log = logging.getLogger(__name__)

CACHE_DIR_NAME = ".cache"


def _list_data_files(data_root: Path) -> list[Path]:
    files: list[Path] = []
    for dir_name in ("source", "masks", "cropped", "cropped_masks"):
        dir_path = data_root / dir_name
        if not dir_path.is_dir():
            continue
        for p in sorted(dir_path.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                files.append(p)
    manifest = data_root / "dataset.json"
    if manifest.is_file():
        files.append(manifest)
    return files


def compute_dataset_signature(data_root: str) -> str:
    root = Path(data_root)
    files = _list_data_files(root)
    lines = []
    for f in files:
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        rel = f.relative_to(root)
        lines.append(f"{rel}:{size}")
    raw = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_cache(data_root: str) -> dict | None:
    sig = compute_dataset_signature(data_root)
    cache_path = Path(data_root) / CACHE_DIR_NAME / f"{sig}.json"
    if not cache_path.is_file():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        log.info("Loaded cache from %s", cache_path)
        return data
    except (json.JSONDecodeError, OSError):
        log.warning("Failed to read cache %s, ignoring", cache_path)
        return None


def save_cache(data_root: str, cache_data: dict) -> None:
    sig = compute_dataset_signature(data_root)
    cache_dir = Path(data_root) / CACHE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{sig}.json"
    cache_path.write_text(
        json.dumps(cache_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Saved cache to %s", cache_path)


def clear_cache(data_root: str) -> None:
    sig = compute_dataset_signature(data_root)
    cache_path = Path(data_root) / CACHE_DIR_NAME / f"{sig}.json"
    if cache_path.exists():
        cache_path.unlink()
        log.info("Cleared cache at %s", cache_path)