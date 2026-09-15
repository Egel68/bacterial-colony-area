from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    data_root: Path = Path("train/data")
    img_size: int = 512
    batch_size: int = 8
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_split: float = 0.2
    num_workers: int = 4
    seed: int = 42
    patience: int = 30
    augment: bool = True
    model_name: str = "unet"
    architectures: tuple[str, ...] = ("unet",)
    run_dir: Path = Path("train/runs")
    device: str = "cuda"
    dashboard: bool = False
    dashboard_port: int = 8765
