import json
import logging
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import TrainingConfig
from .dataset import (
    ColonyPatchDataset,
    GroupedPatchBatchSampler,
    make_datasets,
    make_test_dataset,
)
from .models import get_model

log = logging.getLogger(__name__)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
    n = 0
    amp_enabled = device.type == "cuda"
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    for images, masks in loader:
        images = images.to(device, non_blocking=amp_enabled)
        masks = masks.to(device, non_blocking=amp_enabled)
        optimizer.zero_grad()
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            pred = model(images)
            loss = model.get_loss(pred, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
    amp_enabled = device.type == "cuda"
    for images, masks in loader:
        images = images.to(device, non_blocking=amp_enabled)
        masks = masks.to(device, non_blocking=amp_enabled)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
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
) -> tuple[dict, list, list, Path]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    train_ds, val_ds = make_datasets(
        cfg.data_root,
        cfg.img_size,
        cfg.val_split,
        cfg.seed,
        cfg.augment if cfg.model_name != "mobilenet_v3_small_unet" else False,
    )
    test_ds = make_test_dataset(cfg.data_root, cfg.img_size)
    patch_training = cfg.patch_training or cfg.model_name == "mobilenet_v3_small_unet"
    if patch_training:
        train_ds = ColonyPatchDataset(
            train_ds.records,
            cfg.data_root,
            patch_size=cfg.img_size,
            patches_per_image=cfg.patches_per_image,
            seed=cfg.seed,
            augment=cfg.augment,
            training=True,
        )
        val_ds = ColonyPatchDataset(
            val_ds.records,
            cfg.data_root,
            patch_size=cfg.img_size,
            patches_per_image=(
                max(2, cfg.patches_per_image // 2)
                if cfg.model_name == "mobilenet_v3_small_unet"
                else max(1, cfg.patches_per_image // 2)
            ),
            seed=cfg.seed,
            augment=False,
            training=False,
        )
    if patch_training:
        train_batch_sampler = GroupedPatchBatchSampler(
            train_ds, batch_size=cfg.batch_size, seed=cfg.seed
        )
        val_batch_sampler = GroupedPatchBatchSampler(
            val_ds, batch_size=cfg.batch_size, seed=cfg.seed
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
    log.info(
        "Train patches: %d (%d source images) | Val patches: %d (%d source images)",
        len(train_ds),
        len(train_ds.records) if patch_training else len(train_ds),
        len(val_ds),
        len(val_ds.records) if patch_training else len(val_ds),
    )

    if cfg.model_name == "mobilenet_v3_small_unet":
        pretrained_used = cfg.pretrained and cfg.resume_checkpoint is None
        model = get_model(
            cfg.model_name,
            pretrained=pretrained_used,
        )
    else:
        pretrained_used = False
        model = get_model(cfg.model_name)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 1
    best_iou = 0.0
    best_epoch = -1
    best_val_metrics = {}
    best_state = None
    train_hist = []
    val_hist = []
    resume_mode = "fresh"
    inherited_initialization_provenance = None
    if cfg.resume_checkpoint is not None:
        resume_path = Path(cfg.resume_checkpoint)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_data = torch.load(resume_path, map_location="cpu", weights_only=True)
        if isinstance(resume_data, dict) and "model_state_dict" in resume_data:
            resume_model_state = resume_data["model_state_dict"]
            if resume_data.get("model_name") != cfg.model_name:
                raise ValueError(
                    "Resume checkpoint model does not match training model"
                )
            if resume_data.get("img_size") != cfg.img_size:
                raise ValueError("Resume checkpoint image size does not match")
            expected_resume_config = {
                "max_epochs": cfg.epochs,
                "seed": cfg.seed,
                "batch_size": cfg.batch_size,
                "patches_per_image": cfg.patches_per_image,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "augment": cfg.augment,
            }
            for key, expected in expected_resume_config.items():
                if key in resume_data and resume_data[key] != expected:
                    raise ValueError(
                        f"Resume {key} does not match the saved configuration "
                        f"({resume_data.get(key)!r} != {expected!r})"
                    )
            model.load_state_dict(resume_model_state)
            optimizer.load_state_dict(resume_data["optimizer_state_dict"])
            scheduler.load_state_dict(resume_data["scheduler_state_dict"])
            scaler.load_state_dict(resume_data.get("scaler_state_dict", {}))
            if "torch_rng_state" in resume_data:
                torch.set_rng_state(resume_data["torch_rng_state"])
            if device.type == "cuda" and resume_data.get("cuda_rng_state_all"):
                torch.cuda.set_rng_state_all(resume_data["cuda_rng_state_all"])
            start_epoch = int(resume_data["epoch"]) + 1
            best_iou = float(resume_data["best_iou"])
            best_epoch = int(resume_data["best_epoch"])
            best_val_metrics = dict(resume_data["best_val_metrics"])
            best_state = resume_data["best_state_dict"]
            train_hist = list(resume_data["train_history"])
            val_hist = list(resume_data["validation_history"])
            inherited_initialization_provenance = resume_data.get(
                "initialization_provenance"
            )
            resume_mode = "full_state"
            log.info("Resuming training at epoch %d from %s", start_epoch, resume_path)
        else:
            # Legacy best.pt содержит только веса: продолжить с них с новым optimizer.
            model.load_state_dict(resume_data)
            best_val_metrics = val_epoch(model, val_loader, device)
            best_iou = float(best_val_metrics["iou"])
            best_epoch = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            resume_mode = "warm_start"
            log.info("Warm-starting from weights in %s", resume_path)
        if start_epoch > cfg.epochs:
            raise ValueError(
                f"Resume checkpoint already reached epoch {start_epoch - 1}, "
                f"which meets/exceeds configured maximum {cfg.epochs}"
            )
    initialization_provenance = _initialization_provenance(
        cfg,
        pretrained_used=pretrained_used,
        resume_mode=resume_mode,
        inherited_provenance=inherited_initialization_provenance,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"{cfg.model_name}_{timestamp}"
    run_dir = cfg.run_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, ckpt_dir / "best.pt")

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    compact_patch_model = cfg.model_name == "mobilenet_v3_small_unet"

    for epoch in range(start_epoch, cfg.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        if patch_training:
            train_batch_sampler.set_epoch(epoch - 1)
            val_batch_sampler.set_epoch(epoch - 1)
            train_ds.set_epoch(epoch - 1)

        train_metrics = train_epoch(model, train_loader, optimizer, device, scaler)
        val_metrics = val_epoch(model, val_loader, device)
        scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        train_metrics["lr"] = lr
        train_metrics["seconds"] = elapsed
        train_metrics["epoch"] = epoch
        val_metrics["lr"] = lr
        val_metrics["seconds"] = elapsed
        val_metrics["epoch"] = epoch
        train_hist.append(train_metrics)
        val_hist.append(val_metrics)

        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        log.info(
            "Epoch %d/%d: train_loss=%.4f val_iou=%.4f val_dice=%.4f (%.1fs)",
            epoch,
            cfg.epochs,
            train_metrics["loss"],
            val_metrics["iou"],
            val_metrics["dice"],
            elapsed,
        )

        if best_state is None or val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(best_state, str(ckpt_dir / "best.pt"))
            if not compact_patch_model:
                _export_onnx(model, ckpt_dir / "best.onnx", cfg.img_size)

        current_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        torch.save(current_state, str(ckpt_dir / "last.pt"))
        if not compact_patch_model:
            _export_onnx(model, ckpt_dir / "last.onnx", cfg.img_size)

        resume_state = {
            "model_name": cfg.model_name,
            "img_size": cfg.img_size,
            "max_epochs": cfg.epochs,
            "seed": cfg.seed,
            "batch_size": cfg.batch_size,
            "patches_per_image": cfg.patches_per_image,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "augment": cfg.augment,
            "epoch": epoch,
            "model_state_dict": current_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_iou": best_iou,
            "best_epoch": best_epoch,
            "best_val_metrics": best_val_metrics,
            "best_state_dict": best_state,
            "train_history": train_hist,
            "validation_history": val_hist,
            "initialization_provenance": initialization_provenance,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else [],
        }
        resume_tmp = ckpt_dir / "resume_state.pt.tmp"
        torch.save(resume_state, resume_tmp)
        resume_tmp.replace(ckpt_dir / "resume_state.pt")

        if progress_callback:
            progress_callback(epoch, cfg.epochs, train_metrics, val_metrics, elapsed)

        if epoch - best_epoch > cfg.patience:
            log.info("Early stopping at epoch %d", epoch)
            break

    writer.close()

    if compact_patch_model:
        model = model.to("cpu")
        model.load_state_dict(best_state)
        _export_onnx(model, ckpt_dir / "best.onnx", cfg.img_size)

    summary = {
        "model": cfg.model_name,
        "training_device": str(device),
        "training_accelerator": "cuda" if device.type == "cuda" else "cpu",
        "training_device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "torch_version": torch.__version__,
        "epochs": epoch,
        "best_epoch": best_epoch,
        "best_iou": best_iou,
        **{f"best_val_{key}": value for key, value in best_val_metrics.items()},
        "train_samples": len(train_ds.records) if patch_training else len(train_ds),
        "val_samples": len(val_ds.records) if patch_training else len(val_ds),
        "train_patches": len(train_ds),
        "val_patches": len(val_ds),
        "test_samples": len(test_ds),
        "img_size": cfg.img_size,
        "patch_training": patch_training,
        "pretrained": pretrained_used,
        "pretrained_requested": cfg.pretrained
        if cfg.model_name == "mobilenet_v3_small_unet"
        else None,
        "pretrained_used_for_initial_run": initialization_provenance.get(
            "pretrained_used_for_initial_run"
        ),
        "initialization_source": (
            str(cfg.resume_checkpoint)
            if cfg.resume_checkpoint is not None
            else ("imagenet" if pretrained_used else "random")
        ),
        "resume_checkpoint_sha256": initialization_provenance.get(
            "resume_checkpoint_sha256"
        ),
        "initialization_provenance": initialization_provenance,
        "dataset_root": str(cfg.data_root.resolve()),
        "training_config": {
            "seed": cfg.seed,
            "img_size": cfg.img_size,
            "batch_size": cfg.batch_size,
            "patches_per_image": cfg.patches_per_image,
            "learning_rate": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "patience": cfg.patience,
            "epochs_requested": cfg.epochs,
            "augment": cfg.augment,
            "device_requested": cfg.device,
        },
        "patches_per_image": cfg.patches_per_image if patch_training else None,
        "seconds_per_epoch": [row["seconds"] for row in train_hist],
        "start_epoch": start_epoch,
        "resumed_from": str(cfg.resume_checkpoint)
        if cfg.resume_checkpoint is not None
        else None,
        "resume_mode": resume_mode,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary, train_hist, val_hist, run_dir


def _initialization_provenance(
    cfg: TrainingConfig,
    *,
    pretrained_used: bool,
    resume_mode: str,
    inherited_provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Зафиксировать исходные веса и режим продолжения модели в отчёте."""
    import hashlib

    provenance: dict[str, object] = {
        "pretrained_requested": bool(cfg.pretrained),
        "pretrained_used_for_current_run": bool(pretrained_used),
        "resume_mode": resume_mode,
    }
    if cfg.resume_checkpoint is None:
        provenance["pretrained_used_for_initial_run"] = bool(pretrained_used)
    if inherited_provenance is not None:
        provenance = dict(inherited_provenance)
        provenance["continued_from_full_checkpoint"] = True
        provenance["pretrained_used_for_current_run"] = False
    if cfg.resume_checkpoint is not None:
        checkpoint = Path(cfg.resume_checkpoint).resolve()
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        provenance["resume_checkpoint"] = str(checkpoint)
        provenance["resume_checkpoint_sha256"] = digest.hexdigest()
        provenance["resume_checkpoint_mode"] = resume_mode
        if resume_mode == "warm_start":
            provenance["warm_start_checkpoint"] = str(checkpoint)
            provenance["warm_start_checkpoint_sha256"] = digest.hexdigest()
            provenance["warm_start_lineage"] = "not_recorded_in_legacy_checkpoint"
        elif inherited_provenance is None and cfg.pretrained:
            provenance["pretrained_initialization_source"] = (
                "unknown_ancestor_checkpoint_lineage"
            )
        return provenance

    if pretrained_used:
        from torchvision.models import MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.DEFAULT
        provenance["pretrained_initialization_source"] = weights.url
        provenance["pretrained_used_for_current_run"] = True
        cached_weights = (
            Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
        )
        if cached_weights.is_file():
            digest = hashlib.sha256()
            with cached_weights.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            provenance["pretrained_weights_sha256"] = digest.hexdigest()
    return provenance


def _export_onnx(model: torch.nn.Module, output_path: Path, img_size: int) -> None:
    """Экспорт совместимого с training model API ONNX-графа."""
    if hasattr(model, "to_onnx"):
        model.to_onnx(str(output_path), input_shape=(1, 3, img_size, img_size))
        return
    torch.onnx.export(
        model.eval(),
        torch.randn(1, 3, img_size, img_size, device=next(model.parameters()).device),
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=18,
        external_data=False,
    )
