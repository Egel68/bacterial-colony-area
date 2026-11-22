"""CLI предварительной подготовки COCO-датасета к обучению."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Sequence

from .dataset_archive import pack_training_dataset, unpack_training_archive
from .dataset_distribution import prepare_training_dataset

log = logging.getLogger(__name__)


def prepare_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate and materialize a complete COCO training dataset"
    )
    parser.add_argument(
        "--data-root", default="datasets/22022540", help="COCO source directory"
    )
    parser.add_argument(
        "--output",
        default="datasets/work/22022540-training",
        help="New local prepared dataset directory (must not already exist)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed (default: 42)")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--black-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Materialize full-size black-background variants (default: enabled)",
    )
    args = parser.parse_args(argv)

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(ratio <= 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0):
        parser.error("Train/validation/test ratios must be positive and sum to 1")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        summary = prepare_training_dataset(
            data_root=Path(args.data_root),
            output_dir=Path(args.output),
            seed=args.seed,
            ratios=ratios,
            black_background=args.black_background,
        )
    except Exception as error:
        parser.exit(2, f"ERROR: dataset preparation failed: {error}\n")
    log.info(
        "Подготовка завершена:\n%s", json.dumps(summary, ensure_ascii=False, indent=2)
    )


def pack_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Pack a prepared dataset into a verified .tar.zst archive"
    )
    parser.add_argument("--data-root", default="datasets/work/22022540-training")
    parser.add_argument(
        "--archive", default="datasets/archives/22022540-training.tar.zst"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        summary = pack_training_dataset(Path(args.data_root), Path(args.archive))
    except Exception as error:
        parser.exit(2, f"ERROR: archive creation failed: {error}\n")
    log.info(
        "Архив и sidecar-метаданные созданы:\n%s",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )


def unpack_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Verify and extract a .tar.zst training archive"
    )
    parser.add_argument(
        "--archive", default="datasets/archives/22022540-training.tar.zst"
    )
    parser.add_argument("--output", default="datasets/work/22022540-training")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        summary = unpack_training_archive(Path(args.archive), Path(args.output))
    except Exception as error:
        parser.exit(2, f"ERROR: archive extraction failed: {error}\n")
    log.info(
        "Распаковка завершена:\n%s", json.dumps(summary, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    prepare_main()
