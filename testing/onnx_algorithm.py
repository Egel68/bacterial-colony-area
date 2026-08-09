"""ONNX-адаптер: нейросетевая модель как алгоритм детекции колоний."""

from pathlib import Path

import cv2
import numpy as np

from .interface import BaseDetectionAlgorithm

# Нормализация ImageNet, как в train/predict.py.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class OnnxModelAlgorithm(BaseDetectionAlgorithm):
    """Адаптер ONNX-модели к интерфейсу BaseDetectionAlgorithm.

    Импортирует onnxruntime лениво (в detect), чтобы приложение работало
    без ML-зависимостей, пока не используются нейросетевые алгоритмы.
    """

    def __init__(
        self,
        model_path: str,
        name: str = "",
        description: str = "",
        img_size: int = 512,
        threshold: float = 0.5,
    ):
        self.model_path = str(model_path)
        self._name = name or Path(self.model_path).stem
        self._description = description or (
            f"ONNX-модель: {Path(self.model_path).name}"
        )
        self.img_size = img_size
        self.threshold = threshold
        self._session = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def _get_session(self):
        """Ленивая загрузка ONNX Runtime сессии."""
        if self._session is not None:
            return self._session

        try:
            import onnxruntime
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime не установлен. Установите его в окружении для запуска "
                "нейросетевых моделей: uv pip install onnxruntime"
            ) from e

        self._session = onnxruntime.InferenceSession(self.model_path)
        return self._session

    def detect(self, image: np.ndarray, is_cropped: bool = False) -> np.ndarray:
        session = self._get_session()
        input_name = session.get_inputs()[0].name

        h, w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - MEAN) / STD
        tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :].astype(np.float32)

        raw = session.run(None, {input_name: tensor})[0][0, 0]
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)

        binary = (raw > self.threshold).astype(np.uint8) * 255
        return binary


def scan_bundled_models(models_dir: str = "") -> list["OnnxModelAlgorithm"]:
    """Сканирует папку на *.onnx и возвращает список адаптеров встроенных моделей.

    models_dir по умолчанию — подпапка models/ рядом с пакетом testing.
    """
    if not models_dir:
        default = Path(__file__).resolve().parent.parent / "models"
        models_dir = str(default)
    path = Path(models_dir)
    if not path.is_dir():
        return []

    algos = []
    for model_file in sorted(path.glob("*.onnx")):
        algos.append(
            OnnxModelAlgorithm(
                model_path=str(model_file),
                name=f"NN:{model_file.stem}",
                description=f"Встроенная нейросетевая модель {model_file.name}",
            )
        )
    return algos


def register_bundled_models(
    models_dir: str = "",
) -> list["OnnxModelAlgorithm"]:
    """Регистрирует встроенные модели в реестре алгоритмов. Возвращает зарегистрированных."""
    from .registry import register_algorithm_instance

    algos = scan_bundled_models(models_dir)
    for algo in algos:
        register_algorithm_instance(algo.name, algo)
    return algos
