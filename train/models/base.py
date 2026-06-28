from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
import torch.nn as nn


class BaseSegmenter(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def get_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target)
        sig = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (sig * target).sum(dim=(1, 2, 3))
        union = sig.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1 - (2 * intersection + smooth) / (union + smooth)
        return bce + dice.mean()

    def compute_metrics(
        self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
    ) -> dict[str, float]:
        sig = torch.sigmoid(pred).detach()
        binary = (sig > threshold).float()
        target = target.float()

        smooth = 1e-6
        tp = (binary * target).sum().item()
        fp = (binary * (1 - target)).sum().item()
        fn = ((1 - binary) * target).sum().item()
        tn = ((1 - binary) * (1 - target)).sum().item()

        iou = tp / (tp + fp + fn + smooth)
        dice = 2 * tp / (2 * tp + fp + fn + smooth)
        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)

        loss = self.get_loss(pred, target).item()
        return {
            "loss": loss,
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
        }

    def to_onnx(self, path: str, input_shape: tuple[int, ...] = (1, 3, 512, 512)):
        self.eval()
        dummy = torch.randn(input_shape, device=next(self.parameters()).device)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*dynamic_axes.*")
            torch.onnx.export(
                self,
                dummy,
                path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=18,
            )

    @staticmethod
    def load_onnx(path: str):
        import onnxruntime
        return onnxruntime.InferenceSession(path)

    def preprocess(self, image: np.ndarray, img_size: int = 512) -> np.ndarray:
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = np.transpose(image, (2, 0, 1))
        image = image[None, :, :, :]
        return image.astype(np.float32)
