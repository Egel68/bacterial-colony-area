from pathlib import Path

import cv2
import numpy as np


def batch_predict(
    model_path: str,
    image_paths: list[Path],
    img_size: int = 512,
    threshold: float = 0.5,
) -> list[np.ndarray]:
    import onnxruntime

    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    results = []
    for path in image_paths:
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = np.transpose(image, (2, 0, 1))[None, :, :, :].astype(np.float32)

        pred = session.run(None, {input_name: image})[0][0, 0]
        pred = (pred > threshold).astype(np.uint8) * 255
        results.append(pred)
    return results


def predict_single(
    model_path: str,
    image: np.ndarray,
    img_size: int = 512,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime

    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    h, w = image.shape[:2]

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :].astype(np.float32)

    raw = session.run(None, {input_name: tensor})[0][0, 0]
    raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)

    binary = (raw > threshold).astype(np.uint8) * 255
    raw = (np.clip(raw, 0, 1) * 255).astype(np.uint8)

    return raw, binary
