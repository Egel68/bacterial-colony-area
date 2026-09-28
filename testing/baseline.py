import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

import cv2
import numpy as np
from .cache import SUPPORTED_EXTS
from .registry import get_algorithm_descriptions
from .runner import compute_detailed_metrics
from .scheduler import execute_pipeline
from .telemetry import TelemetryCollector

log = logging.getLogger(__name__)


class BaselineSample:
    __test__ = False

    def __init__(
        self,
        name: str,
        image_path: Path,
        mask_path: Path,
        variant: Literal["source", "cropped"],
    ):
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
        files = [
            p
            for p in source_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]
        for f in files:
            mask_candidate = source_dir / f"{f.stem}_mask{f.suffix}"
            if mask_candidate.exists():
                return "importer"
    raise RuntimeError(f"No valid image-mask pairs found in {root}")


class BaselineDataset:
    __test__ = False

    def __init__(self, root: str | Path, sample_limit: int | None = None):
        if sample_limit is not None and sample_limit <= 0:
            raise ValueError("sample_limit must be positive")
        self.root = Path(root)
        self.sample_limit = sample_limit
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
        self._apply_sample_limit()

    def _apply_sample_limit(self) -> None:
        if self.sample_limit is None:
            return
        source_names = sorted(
            sample.name for sample in self._samples if sample.variant == "source"
        )[: self.sample_limit]
        selected = set(source_names)
        self._samples = [
            sample
            for sample in self._samples
            if sample.name in selected
            or sample.name.removesuffix("_cropped") in selected
        ]

    def _load_legacy(self):
        source_dir = self.root / "source"
        masks_dir = self.root / "masks"
        cropped_dir = self.root / "cropped"
        cropped_masks_dir = self.root / "cropped_masks"

        source_count = 0
        for source_path in sorted(source_dir.iterdir()):
            if (
                not source_path.is_file()
                or source_path.suffix.lower() not in SUPPORTED_EXTS
            ):
                continue
            stem = source_path.stem
            ext = source_path.suffix
            mask_path = masks_dir / f"{stem}_mask{ext}"
            if not mask_path.exists():
                continue
            source_count += 1
            self._samples.append(
                BaselineSample(
                    name=stem,
                    image_path=source_path,
                    mask_path=mask_path,
                    variant="source",
                )
            )
            if self.sample_limit is not None and source_count >= self.sample_limit:
                break

            cropped_path = cropped_dir / f"{stem}_cropped{ext}"
            cropped_mask_path = cropped_masks_dir / f"{stem}_cropped_mask{ext}"
            if cropped_path.exists() and cropped_mask_path.exists():
                self._samples.append(
                    BaselineSample(
                        name=f"{stem}_cropped",
                        image_path=cropped_path,
                        mask_path=cropped_mask_path,
                        variant="cropped",
                    )
                )

    def _load_importer(self):
        source_dir = self.root / "source"
        cropped_dir = self.root / "cropped"

        source_count = 0
        selected_source_names: set[str] = set()
        for source_path in sorted(source_dir.iterdir()):
            if (
                not source_path.is_file()
                or source_path.suffix.lower() not in SUPPORTED_EXTS
                or source_path.stem.endswith("_mask")
            ):
                continue
            stem = source_path.stem
            ext = source_path.suffix
            mask_path = source_dir / f"{stem}_mask{ext}"
            if not mask_path.exists():
                continue
            source_count += 1
            self._samples.append(
                BaselineSample(
                    name=stem,
                    image_path=source_path,
                    mask_path=mask_path,
                    variant="source",
                )
            )
            selected_source_names.add(stem)
            if self.sample_limit is not None and source_count >= self.sample_limit:
                break

        if cropped_dir.is_dir() and (self.sample_limit is None or source_count):
            for cropped_path in sorted(cropped_dir.iterdir()):
                if (
                    not cropped_path.is_file()
                    or cropped_path.suffix.lower() not in SUPPORTED_EXTS
                ):
                    continue
                stem = cropped_path.stem
                source_stem = stem.removesuffix("_cropped")
                if (
                    self.sample_limit is not None
                    and source_stem not in selected_source_names
                ):
                    continue
                ext = cropped_path.suffix
                mask_path = cropped_dir / f"{stem}_mask{ext}"
                if not mask_path.exists():
                    continue
                self._samples.append(
                    BaselineSample(
                        name=stem,
                        image_path=cropped_path,
                        mask_path=mask_path,
                        variant="cropped",
                    )
                )

    def _load_manifest(self):
        import json

        manifest_path = self.root / "dataset.json"
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))

        selected_source_ids: set[str] = set()
        source_count = 0
        entries = raw.get("samples", [])
        for entry in sorted(entries, key=lambda item: str(item.get("id", ""))):
            kind = entry.get("kind", "source")
            if kind != "source":
                continue
            image_rel = Path(entry["image"])
            mask_rel = Path(entry["mask"])
            image_path = self.root / image_rel
            mask_path = self.root / mask_rel
            if not image_path.is_file() or not mask_path.is_file():
                continue
            self._samples.append(
                BaselineSample(
                    name=entry["id"],
                    image_path=image_path,
                    mask_path=mask_path,
                    variant=kind,
                )
            )
            selected_source_ids.add(entry["id"])
            source_count += 1
            if self.sample_limit is not None and source_count >= self.sample_limit:
                break

        for entry in sorted(entries, key=lambda item: str(item.get("id", ""))):
            if entry.get("kind") != "cropped":
                continue
            source_id = str(entry.get("id", "")).removesuffix("_cropped")
            if self.sample_limit is not None and source_id not in selected_source_ids:
                continue
            image_rel = Path(entry["image"])
            mask_rel = Path(entry["mask"])
            image_path = self.root / image_rel
            mask_path = self.root / mask_rel
            if not image_path.is_file() or not mask_path.is_file():
                continue
            self._samples.append(
                BaselineSample(
                    name=entry["id"],
                    image_path=image_path,
                    mask_path=mask_path,
                    variant="cropped",
                )
            )

    @property
    def samples(self) -> list[BaselineSample]:
        return list(self._samples)

    def samples_by_variant(
        self, variant: Literal["source", "cropped"]
    ) -> list[BaselineSample]:
        return [s for s in self._samples if s.variant == variant]

    def count_source(self) -> int:
        return sum(1 for s in self._samples if s.variant == "source")

    def count_cropped(self) -> int:
        return sum(1 for s in self._samples if s.variant == "cropped")

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[BaselineSample]:
        return iter(self._samples)


def run_baseline(
    dataset: BaselineDataset,
    use_cropped: bool = True,
    cache: dict | None = None,
    workers: int | None = None,
    use_cache: bool = False,
    batch_size: int = 8,
    memory_budget: int | None = None,
    telemetry: TelemetryCollector | None = None,
) -> dict:
    algo_names = [
        "ClassicDefault",
        "ClassicHighSensitivity",
        "ClassicSolidFill",
        "ClassicLowSensitivity",
    ]
    desc_map = dict(get_algorithm_descriptions())

    result: dict = {
        "dataset_name": dataset.root.name,
        "image_count": {
            "source": dataset.count_source(),
            "cropped": dataset.count_cropped() if use_cropped else 0,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithms": [],
    }

    if cache is None:
        cache = {}
    if dataset.sample_limit is not None and cache:
        log.info(
            "Ignoring aggregate cache for limited dataset (%d samples)",
            dataset.sample_limit,
        )
        cache = {}
    if cache:
        log.info(
            "Cache provided with %d algorithm(s): %s", len(cache), list(cache.keys())
        )

    uncached_names = [n for n in algo_names if n not in cache]

    for cached_name in algo_names:
        if cached_name in cache:
            log.info("Skipping %s (cached)", cached_name)
            result["algorithms"].append(cache[cached_name])

    collected: dict[str, dict[str, dict[str, float]]] = {}
    if uncached_names:
        results = execute_pipeline(
            dataset,
            uncached_names,
            workers=workers,
            batch_size=batch_size,
            memory_budget=memory_budget,
            use_cache=use_cache,
            telemetry=telemetry,
            use_cropped=use_cropped,
        )
        for algo_name, samples in results.items():
            collected[algo_name] = {
                (sample_name, variant): metrics
                for sample_name, variants in samples.items()
                for variant, metrics in variants.items()
            }

    for algo_name in uncached_names:
        metrics_by_sample = collected.get(algo_name, {})
        source_metrics = [
            metrics_by_sample[(sample.name, "source")]
            for sample in dataset
            if (sample.name, "source") in metrics_by_sample
        ]
        cropped_metrics = [
            metrics_by_sample[(sample.name, "cropped")]
            for sample in dataset
            if use_cropped and (sample.name, "cropped") in metrics_by_sample
        ]
        entry: dict = {
            "name": algo_name,
            "description": desc_map.get(algo_name, ""),
            "source": compute_detailed_metrics(source_metrics),
        }
        entry["source"]["num_samples"] = len(source_metrics)

        if cropped_metrics:
            entry["cropped"] = compute_detailed_metrics(cropped_metrics)
            entry["cropped"]["num_samples"] = len(cropped_metrics)

        cache[algo_name] = entry
    result["algorithms"] = [cache[name] for name in algo_names if name in cache]

    return result


def export_json(result: dict, output_path: str) -> None:
    import json

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Baseline report written to %s", out)
