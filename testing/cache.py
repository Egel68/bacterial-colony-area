import hashlib
import json
import logging
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)

CACHE_DIR_NAME = ".cache"
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


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


PRED_CACHE_DIR = "preds"


class PredictionCache:
    """Content-addressable cache for prediction masks.

    Stores PNG masks in {data_root}/.cache/preds/{algo_name}/{hash}.png.
    The hash is sha256(image_bytes + algo_name).
    """

    def __init__(self, data_root: str | Path):
        self.root = Path(data_root) / CACHE_DIR_NAME / PRED_CACHE_DIR

    def _hash(
        self,
        image_path: str,
        algo_name: str,
        params: object | None = None,
    ) -> str:
        h = hashlib.sha256()
        h.update(algo_name.encode("utf-8"))
        h.update(self._params_bytes(params))
        try:
            with open(image_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    h.update(chunk)
        except OSError:
            return ""
        return h.hexdigest()

    @staticmethod
    def _params_bytes(params: object | None) -> bytes:
        if params is None:
            return b""
        if isinstance(params, bytes):
            return params
        try:
            return json.dumps(params, sort_keys=True, default=str).encode("utf-8")
        except (TypeError, ValueError):
            return repr(params).encode("utf-8")

    @classmethod
    def _array_hash(
        cls,
        image: np.ndarray,
        algo_name: str,
        params: object | None = None,
    ) -> str:
        h = hashlib.sha256()
        h.update(algo_name.encode("utf-8"))
        h.update(cls._params_bytes(params))
        h.update(str(image.dtype).encode("ascii"))
        h.update(repr(image.shape).encode("ascii"))
        h.update(np.ascontiguousarray(image).tobytes())
        return h.hexdigest()

    def key(
        self,
        image_path: str,
        algo_name: str,
        params: object | None = None,
    ) -> str | None:
        """Return cache key (hash) or None if image can't be read."""
        return self._hash(image_path, algo_name, params)

    def get(
        self,
        image_path: str,
        algo_name: str,
        params: object | None = None,
    ) -> np.ndarray | None:
        """Return cached mask as uint8 array, or None if not found."""
        import cv2
        digest = self._hash(image_path, algo_name, params)
        if not digest:
            return None
        cache_path = self.root / algo_name / f"{digest}.png"
        if not cache_path.is_file():
            return None
        mask = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
        return mask if mask is not None else None

    def get_array(
        self,
        image: np.ndarray,
        algo_name: str,
        params: object | None = None,
    ) -> np.ndarray | None:
        """Return a prediction using a digest of an already loaded image."""
        import cv2

        digest = self._array_hash(image, algo_name, params)
        cache_path = self.root / algo_name / f"{digest}.png"
        if not cache_path.is_file():
            return None
        mask = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
        return mask if mask is not None else None

    def put(
        self,
        image_path: str,
        algo_name: str,
        mask: np.ndarray,
        params: object | None = None,
    ) -> None:
        """Store mask in cache."""
        import cv2
        digest = self._hash(image_path, algo_name, params)
        if not digest:
            return
        cache_dir = self.root / algo_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{digest}.png"
        if not cache_path.exists():
            cv2.imwrite(str(cache_path), mask)

    def put_array(
        self,
        image: np.ndarray,
        algo_name: str,
        mask: np.ndarray,
        params: object | None = None,
    ) -> None:
        """Store a prediction without reopening the source image."""
        import cv2

        digest = self._array_hash(image, algo_name, params)
        cache_dir = self.root / algo_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{digest}.png"
        if not cache_path.exists():
            cv2.imwrite(str(cache_path), mask)

    def clear(self, algo_name: str | None = None) -> None:
        """Clear cache for one algorithm or all."""
        target = self.root / algo_name if algo_name else self.root
        if target.is_dir():
            import shutil
            shutil.rmtree(target)
            log.info("Cleared prediction cache: %s", target)
