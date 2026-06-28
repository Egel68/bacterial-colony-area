import argparse
import json
import threading
from pathlib import Path

from .config import TrainingConfig
from .dataset import make_datasets
from .models import get_model, list_models
from .train import run_training
from .reporter import generate_report


def _train(args):
    cfg = TrainingConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        dashboard=args.dashboard,
        dashboard_port=args.dashboard_port,
    )

    if args.dashboard:
        from .dashboard.server import run_server, update
        server = threading.Thread(target=run_server, args=(cfg.dashboard_port,), daemon=True)
        server.start()
        print(f"Dashboard: http://127.0.0.1:{cfg.dashboard_port}")
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
    print(f"Report: {report_path}")


def _list_models(_):
    for name in list_models():
        print(f"  {name}")


def _dataset_info(_):
    cfg = TrainingConfig()
    train_ds, val_ds = make_datasets(cfg.data_root, cfg.img_size, cfg.val_split, cfg.seed, cfg.augment)
    print(f"Train: {len(train_ds)} samples")
    print(f"Val:   {len(val_ds)} samples")


def main():
    parser = argparse.ArgumentParser(description="Colony segmentation training")
    sub = parser.add_subparsers(dest="command")

    train_parser = sub.add_parser("train", help="Run training")
    train_parser.add_argument("--model", default="unet", help="Model name")
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--img-size", type=int, default=512)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--dashboard", action="store_true", help="Enable web dashboard")
    train_parser.add_argument("--dashboard-port", type=int, default=8765)

    sub.add_parser("list-models", help="List available models")
    sub.add_parser("dataset-info", help="Show dataset info")

    args = parser.parse_args()
    if args.command == "train":
        _train(args)
    elif args.command == "list-models":
        _list_models(args)
    elif args.command == "dataset-info":
        _dataset_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
