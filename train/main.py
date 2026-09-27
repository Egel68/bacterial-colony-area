import argparse
import math
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Sequence

from .config import TrainingConfig
from .dataset import make_datasets
from .models import list_models
from .reporter import generate_report
from .train import run_training
from utils.logging import setup_logging

log = logging.getLogger(__name__)


def _train(args):
    cfg = TrainingConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        dashboard=args.dashboard,
        dashboard_port=args.dashboard_port,
        data_root=args.data_root,
    )

    if args.dashboard:
        from .dashboard.server import run_server, update

        server = threading.Thread(
            target=run_server, args=(cfg.dashboard_port,), daemon=True
        )
        server.start()
        log.info("Dashboard: http://127.0.0.1:%d", cfg.dashboard_port)

        try:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tensorboard.main",
                    "--logdir",
                    str(cfg.run_dir),
                    "--port",
                    "6006",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log.info("TensorBoard: http://127.0.0.1:6006")
        except Exception:
            log.warning("TensorBoard: не удалось запустить (возможно порт занят)")

        progress_callback = update
    else:
        from rich.progress import (
            BarColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
        )

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        task_id = None

        def _rich_callback(epoch, total, train_metrics, val_metrics, elapsed):
            nonlocal task_id
            if task_id is None:
                task_id = progress.add_task(f"Epoch {epoch}/{total}", total=total)
            progress.update(
                task_id,
                advance=1,
                description=f"Epoch {epoch}/{total} | loss={val_metrics['loss']:.4f} iou={val_metrics['iou']:.4f} dice={val_metrics['dice']:.4f}",
            )

        progress_callback = _rich_callback
        progress.start()

    try:
        summary, train_hist, val_hist, run_dir = run_training(cfg, progress_callback)
    finally:
        if not args.dashboard:
            progress.stop()

    summary["train_log"] = train_hist
    summary["val_log"] = val_hist

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    report_path = run_dir / "report.html"
    generate_report(summary, train_hist, val_hist, report_path)
    log.info("Report: %s", report_path)


def _train_compare(args):
    from .compare import run_train_compare

    cfg = TrainingConfig(
        model_name=args.architectures.split(",")[0],
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        architectures=tuple(a.strip() for a in args.architectures.split(",")),
        dashboard=args.dashboard,
        dashboard_port=args.dashboard_port,
    )
    run_train_compare(
        cfg.architectures,
        cfg,
        data_root=args.data_root,
        eval_root=args.eval_root,
    )


def _list_models(_):
    for name in list_models():
        log.info("  %s", name)


def _dataset_info(_):
    cfg = TrainingConfig()
    train_ds, val_ds = make_datasets(
        cfg.data_root, cfg.img_size, cfg.val_split, cfg.seed, cfg.augment
    )
    log.info("Train: %d samples", len(train_ds))
    log.info("Val:   %d samples", len(val_ds))


def _train_colony(args, parser):
    from .training_pipeline import run_colony_training

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(ratio <= 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0):
        parser.error("Train/validation/test ratios must be positive and sum to 1")
    if args.epochs <= 0 or args.batch_size <= 0 or args.patches_per_image <= 0:
        parser.error("epochs, batch-size and patches-per-image must be positive")
    if args.img_size != 512:
        parser.error(
            "Native-resolution patch training currently requires --img-size 512"
        )

    run_colony_training(
        data_root=Path(args.data_root),
        output_root=Path(args.output),
        seed=args.seed,
        ratios=ratios,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patches_per_image=args.patches_per_image,
        img_size=args.img_size,
        pretrained=not args.no_pretrained,
        model_promotion_path=(Path(args.promote_model) if args.promote_model else None),
        allow_overwrite=args.allow_overwrite,
        cpu_benchmark=args.cpu_benchmark,
        resume_run=Path(args.resume_run) if args.resume_run else None,
    )


def main(argv: Sequence[str] | None = None):
    setup_logging(logging.INFO)
    parser = argparse.ArgumentParser(description="Colony segmentation training")
    sub = parser.add_subparsers(dest="command")

    train_parser = sub.add_parser("train", help="Run training")
    train_parser.add_argument("--model", default="unet", help="Model name")
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--img-size", type=int, default=512)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument(
        "--data-root", default="train/data", help="Training data root"
    )
    train_parser.add_argument(
        "--dashboard", action="store_true", help="Enable web dashboard"
    )
    train_parser.add_argument("--dashboard-port", type=int, default=8765)

    compare_parser = sub.add_parser(
        "train-compare", help="Train and compare architectures"
    )
    compare_parser.add_argument(
        "--architectures", default="unet,unet_small", help="Comma-separated model names"
    )
    compare_parser.add_argument(
        "--data-root", default=Path("train/data"), help="Training data root"
    )
    compare_parser.add_argument(
        "--eval-root",
        default=Path("test_images"),
        help="Test pairs root for evaluation",
    )
    compare_parser.add_argument("--epochs", type=int, default=200)
    compare_parser.add_argument("--batch-size", type=int, default=8)
    compare_parser.add_argument("--img-size", type=int, default=512)
    compare_parser.add_argument("--lr", type=float, default=1e-3)
    compare_parser.add_argument(
        "--dashboard", action="store_true", help="Enable web dashboard"
    )
    compare_parser.add_argument("--dashboard-port", type=int, default=8765)

    sub.add_parser("list-models", help="List available models")
    sub.add_parser("dataset-info", help="Show dataset info")

    colony_parser = sub.add_parser(
        "train-colony", help="Prepare, train and evaluate the compact COCO segmenter"
    )
    colony_parser.add_argument(
        "--data-root", required=True, help="COCO source directory"
    )
    colony_parser.add_argument(
        "--output", required=True, help="Output root for dataset and run"
    )
    colony_parser.add_argument("--seed", type=int, default=42)
    colony_parser.add_argument("--train-ratio", type=float, default=0.70)
    colony_parser.add_argument("--val-ratio", type=float, default=0.15)
    colony_parser.add_argument("--test-ratio", type=float, default=0.15)
    colony_parser.add_argument("--epochs", type=int, default=100)
    colony_parser.add_argument("--batch-size", type=int, default=8)
    colony_parser.add_argument("--patches-per-image", type=int, default=4)
    colony_parser.add_argument("--img-size", type=int, default=512)
    colony_parser.add_argument("--no-pretrained", action="store_true")
    colony_parser.add_argument(
        "--promote-model", help="Optional explicit ONNX promotion path"
    )
    colony_parser.add_argument("--allow-overwrite", action="store_true")
    colony_parser.add_argument("--cpu-benchmark", action="store_true")
    colony_parser.add_argument(
        "--resume-run",
        help=(
            "Continue an existing run from its resume state; older runs warm-start "
            "from last.pt or best.pt"
        ),
    )

    args = parser.parse_args(argv)
    if args.command == "train":
        _train(args)
    elif args.command == "train-compare":
        _train_compare(args)
    elif args.command == "list-models":
        _list_models(args)
    elif args.command == "dataset-info":
        _dataset_info(args)
    elif args.command == "train-colony":
        _train_colony(args, parser)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
