"""Переэкспорт best.pt в самодостаточный single-file ONNX (без внешних данных)."""

import argparse
import warnings
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Re-export best.pt to a single-file ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument("--model", default="unet", help="Model architecture name")
    parser.add_argument("--img-size", type=int, default=512, help="Input spatial size")
    args = parser.parse_args()

    from .models import get_model

    model = get_model(args.model)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    input_shape = (1, 3, args.img_size, args.img_size)
    dummy = torch.randn(input_shape)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*dynamic_axes.*")
        torch.onnx.export(
            model,
            dummy,
            args.output,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=18,
            external_data=False,
        )
    print(f"Exported single-file ONNX: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()