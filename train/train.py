import json
import time
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import TrainingConfig
from .dataset import make_datasets
from .models import get_model


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
    n = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        pred = model(images)
        loss = model.get_loss(pred, masks)
        loss.backward()
        optimizer.step()
        metrics = model.compute_metrics(pred, masks)
        total_loss += loss.item()
        for k in total_metrics:
            total_metrics[k] += metrics[k]
        n += 1
    avg = {k: v / n for k, v in total_metrics.items()}
    avg["loss"] = total_loss / n
    return avg


@torch.no_grad()
def val_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
    n = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        pred = model(images)
        metrics = model.compute_metrics(pred, masks)
        total_loss += metrics["loss"]
        for k in total_metrics:
            total_metrics[k] += metrics[k]
        n += 1
    avg = {k: v / n for k, v in total_metrics.items()}
    avg["loss"] = total_loss / n
    return avg


def run_training(
    cfg: TrainingConfig,
    progress_callback=None,
) -> tuple[dict, list, list]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds = make_datasets(cfg.data_root, cfg.img_size, cfg.val_split, cfg.seed, cfg.augment)
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = get_model(cfg.model_name)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.model_name}_{timestamp}"
    run_dir = cfg.run_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    best_iou = 0.0
    best_epoch = -1
    best_state = None
    train_hist = []
    val_hist = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()

        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = val_epoch(model, val_loader, device)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        train_metrics["lr"] = lr
        val_metrics["lr"] = lr
        train_hist.append(train_metrics)
        val_hist.append(val_metrics)

        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, str(ckpt_dir / "best.pt"))

        torch.save(model.state_dict(), str(ckpt_dir / "last.pt"))

        if progress_callback:
            progress_callback(epoch, cfg.epochs, train_metrics, val_metrics, elapsed)

        if epoch - best_epoch > cfg.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to_onnx(str(ckpt_dir / "model.onnx"))

    writer.close()

    summary = {
        "model": cfg.model_name,
        "epochs": epoch,
        "best_epoch": best_epoch,
        "best_iou": best_iou,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "img_size": cfg.img_size,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary, train_hist, val_hist
