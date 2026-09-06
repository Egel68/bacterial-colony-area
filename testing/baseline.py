import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal, Optional

import cv2
import numpy as np

from .metrics import compute_segmentation_metrics
from .registry import get_algorithm, list_algorithms, get_algorithm_descriptions
from .runner import _mean_metrics

log = logging.getLogger(__name__)

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class BaselineSample:
    __test__ = False

    def __init__(self, name: str, image_path: Path, mask_path: Path, variant: Literal["source", "cropped"]):
        self.name = name
        self.image_path = image_path
        self.mask_path = mask_path
        self.variant = variant

    def load_image(self) -> np.ndarray:
        return cv2.imread(str(self.image_path))

    def load_mask(self) -> np.ndarray:
        return cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)


DetectedStructure = Literal["manifest", "legacy", "importer"]


def _detect_structure(root: Path) -> DetectedStructure:
    if (root / "dataset.json").is_file():
        return "manifest"
    if (root / "source").is_dir() and (root / "masks").is_dir():
        return "legacy"
    if (root / "source").is_dir():
        source_dir = root / "source"
        files = [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
        for f in files:
            mask_candidate = source_dir / f"{f.stem}_mask{f.suffix}"
            if mask_candidate.exists():
                return "importer"
    raise RuntimeError(f"No valid image-mask pairs found in {root}")


class BaselineDataset:
    __test__ = False

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._structure = _detect_structure(self.root)
        self._samples: list[BaselineSample] = []
        self._load_all()

    @property
    def structure(self) -> DetectedStructure:
        return self._structure

    def _load_all(self):
        if self._structure == "legacy":
            self._load_legacy()
        elif self._structure == "importer":
            self._load_importer()
        elif self._structure == "manifest":
            self._load_manifest()

    def _load_legacy(self):
        source_dir = self.root / "source"
        masks_dir = self.root / "masks"
        cropped_dir = self.root / "cropped"
        cropped_masks_dir = self.root / "cropped_masks"

        for source_path in sorted(source_dir.iterdir()):
            if not source_path.is_file() or source_path.suffix.lower() not in SUPPORTED_EXTS:
                continue
            stem = source_path.stem
            ext = source_path.suffix
            mask_path = masks_dir / f"{stem}_mask{ext}"
            if not mask_path.exists():
                continue
            self._samples.append(BaselineSample(
                name=stem,
                image_path=source_path,
                mask_path=mask_path,
                variant="source",
            ))

            cropped_path = cropped_dir / f"{stem}_cropped{ext}"
            cropped_mask_path = cropped_masks_dir / f"{stem}_cropped_mask{ext}"
            if cropped_path.exists() and cropped_mask_path.exists():
                self._samples.append(BaselineSample(
                    name=f"{stem}_cropped",
                    image_path=cropped_path,
                    mask_path=cropped_mask_path,
                    variant="cropped",
                ))

    def _load_importer(self):
        source_dir = self.root / "source"
        cropped_dir = self.root / "cropped"

        for source_path in sorted(source_dir.iterdir()):
            if not source_path.is_file() or source_path.suffix.lower() not in SUPPORTED_EXTS:
                continue
            stem = source_path.stem
            ext = source_path.suffix
            mask_path = source_dir / f"{stem}_mask{ext}"
            if not mask_path.exists():
                continue
            self._samples.append(BaselineSample(
                name=stem,
                image_path=source_path,
                mask_path=mask_path,
                variant="source",
            ))

        if cropped_dir.is_dir():
            for cropped_path in sorted(cropped_dir.iterdir()):
                if not cropped_path.is_file() or cropped_path.suffix.lower() not in SUPPORTED_EXTS:
                    continue
                stem = cropped_path.stem
                ext = cropped_path.suffix
                mask_path = cropped_dir / f"{stem}_mask{ext}"
                if not mask_path.exists():
                    continue
                self._samples.append(BaselineSample(
                    name=stem,
                    image_path=cropped_path,
                    mask_path=mask_path,
                    variant="cropped",
                ))

    def _load_manifest(self):
        import json

        manifest_path = self.root / "dataset.json"
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))

        for entry in raw.get("samples", []):
            kind = entry.get("kind", "source")
            if kind not in ("source", "cropped"):
                continue
            image_rel = Path(entry["image"])
            mask_rel = Path(entry["mask"])
            image_path = self.root / image_rel
            mask_path = self.root / mask_rel
            if not image_path.is_file() or not mask_path.is_file():
                continue
            self._samples.append(BaselineSample(
                name=entry["id"],
                image_path=image_path,
                mask_path=mask_path,
                variant=kind,
            ))

    @property
    def samples(self) -> list[BaselineSample]:
        return list(self._samples)

    def samples_by_variant(self, variant: Literal["source", "cropped"]) -> list[BaselineSample]:
        return [s for s in self._samples if s.variant == variant]

    def count_source(self) -> int:
        return sum(1 for s in self._samples if s.variant == "source")

    def count_cropped(self) -> int:
        return sum(1 for s in self._samples if s.variant == "cropped")

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[BaselineSample]:
        return iter(self._samples)


def run_baseline(dataset: BaselineDataset, use_cropped: bool = True, cache: dict | None = None) -> dict:
    algo_names = [
        "ClassicDefault",
        "ClassicHighSensitivity",
        "ClassicSolidFill",
        "ClassicLowSensitivity",
    ]
    desc_map = dict(get_algorithm_descriptions())

    result: dict = {
        "dataset_name": dataset.root.name,
        "image_count": {"source": dataset.count_source(), "cropped": dataset.count_cropped() if use_cropped else 0},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithms": [],
    }

    if cache is None:
        cache = {}
    if cache:
        log.info("Cache provided with %d algorithm(s): %s", len(cache), list(cache.keys()))

    for algo_name in algo_names:
        cached_entry = cache.get(algo_name)
        if cached_entry is not None:
            log.info("Skipping %s (cached)", algo_name)
            entry = cached_entry
        else:
            algo = get_algorithm(algo_name)
            source_metrics_list = []
            cropped_metrics_list = []

            try:
                from tqdm import tqdm as _tqdm
                iterator = _tqdm(dataset, desc=algo_name, unit="img")
            except ImportError:
                log.info("Running: %s", algo_name)
                iterator = dataset

            for sample in iterator:
                if sample.variant == "source":
                    img = sample.load_image()
                    gt = sample.load_mask()
                    pred = algo.detect(img, is_cropped=False)
                    source_metrics_list.append(compute_segmentation_metrics(pred, gt))

                if sample.variant == "cropped" and use_cropped:
                    img = sample.load_image()
                    gt = sample.load_mask()
                    pred = algo.detect(img, is_cropped=True)
                    cropped_metrics_list.append(compute_segmentation_metrics(pred, gt))

            entry: dict = {
                "name": algo_name,
                "description": desc_map.get(algo_name, ""),
                "source": _mean_metrics(source_metrics_list),
            }
            entry["source"]["num_samples"] = len(source_metrics_list)

            if use_cropped and cropped_metrics_list:
                entry["cropped"] = _mean_metrics(cropped_metrics_list)
                entry["cropped"]["num_samples"] = len(cropped_metrics_list)

            cache[algo_name] = entry

        result["algorithms"].append(entry)

    return result


def export_json(result: dict, output_path: str) -> None:
    import json

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Baseline report written to %s", out)