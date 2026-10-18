"""Эталонные (reference) фикстуры и baseline бинарных масок/метрик.

Используется задачей 1.2 change `optimize-image-processing-ui-responsiveness`:
детерминированные синтетические кадры (ordinary / boundary / largest,
варианты source / cropped), захват бинарных масок классических алгоритмов
детекции и их метрик относительно ground-truth масок фикстур.

Baseline служит страховкой для последующих оптимизаций (задачи 3.2–3.3):
до/после оптимизации маски обязаны совпадать побитово, а метрики — в допуске
`1e-6`.

Захват baseline (перезапись эталона после легитимного изменения окружения):

    UV_PROJECT_ENVIRONMENT=.venv-dev uv run python tests/reference_fixtures.py --write

Проверка текущего кода против зафиксированного baseline:

    UV_PROJECT_ENVIRONMENT=.venv-dev uv run python tests/reference_fixtures.py --check
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Literal, Tuple

import cv2
import numpy as np

BASELINE_DIR = Path(__file__).resolve().parent / "baseline"
BASELINE_JSON = BASELINE_DIR / "reference_masks.json"
MASKS_DIR = BASELINE_DIR / "masks"

SCHEMA_VERSION = 1

# Классические детерминированные алгоритмы: путь, который оптимизируют задачи группы 3.
ALGORITHMS: Tuple[str, ...] = (
    "ClassicDefault",
    "ClassicHighSensitivity",
    "ClassicSolidFill",
    "ClassicLowSensitivity",
)

Variant = Literal["source", "cropped"]


@dataclass(frozen=True)
class ReferenceFixture:
    """Синтетическая эталонная пара «изображение + ground-truth маска»."""

    name: str
    category: Literal["ordinary", "boundary", "largest"]
    variant: Variant
    image: np.ndarray
    ground_truth: np.ndarray

    @property
    def is_cropped(self) -> bool:
        return self.variant == "cropped"


# ----------------------------------------------------------------------------
# Генерация синтетических кадров (детерминированная, без глобального RNG)
# ----------------------------------------------------------------------------


def _base_canvas(size: Tuple[int, int], rng: np.random.RandomState) -> np.ndarray:
    """Тёмное поле с лёгким детерминированным шумом (как на dark-field фото)."""
    h, w = size
    noise = rng.randint(26, 43, (h, w)).astype(np.uint8)
    return cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)


def _draw_dish(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    rng: np.random.RandomState,
) -> None:
    """Чашка Петри: поле со слабым шумом и яркий блик по краю.

    Яркий край — то, что ищет `ColonyDetector.detect_petri_dish`
    (порог яркости 200). Толщина блика ограничена абсолютными пикселями,
    чтобы кодировка детекции не менялась с размером кадра.
    """
    h, w = image.shape[:2]
    cx, cy = center
    noise = rng.randint(114, 123, (h, w)).astype(np.uint8)
    dish_field = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    disk = np.zeros((h, w), np.uint8)
    cv2.circle(disk, (cx, cy), radius, 255, -1)
    image[disk > 0] = dish_field[disk > 0]
    rim = int(np.clip(radius // 80, 8, 22))
    cv2.circle(image, (cx, cy), radius, (240, 240, 240), rim)


def _draw_colonies(
    image: np.ndarray,
    gt: np.ndarray,
    specs: Iterable[Tuple[Tuple[int, int], int]],
) -> None:
    """Колонии: светлые диски; GT-маска — те же диски без размытия."""
    for (x, y), r in specs:
        val = int(170 + (r * 7) % 26)
        cv2.circle(image, (x, y), r, (val, val, val), -1)
        cv2.circle(gt, (x, y), r, 255, -1)


def _scatter(
    rng: np.random.RandomState,
    center: Tuple[int, int],
    radius: int,
    count: int,
    rmin: int,
    rmax: int,
    dist_max: float = 0.72,
) -> list:
    """Случайные, но детерминированные позиции колоний внутри чашки."""
    cx, cy = center
    specs = []
    for _ in range(count):
        ang = rng.uniform(0.0, 2.0 * np.pi)
        dist = rng.uniform(0.05, dist_max) * radius
        x = int(round(cx + dist * np.cos(ang)))
        y = int(round(cy + dist * np.sin(ang)))
        r = int(rng.randint(rmin, rmax + 1))
        specs.append(((x, y), r))
    return specs


def _ring_positions(
    rng: np.random.RandomState,
    center: Tuple[int, int],
    radius: int,
    count: int,
    dist: float,
    rmin: int,
    rmax: int,
) -> list:
    """Колонии на фиксированном расстоянии от центра (граница ROI/край чашки)."""
    cx, cy = center
    specs = []
    for i in range(count):
        ang = rng.uniform(0.0, 2.0 * np.pi) + i * (2.0 * np.pi / count)
        x = int(round(cx + dist * radius * np.cos(ang)))
        y = int(round(cy + dist * radius * np.sin(ang)))
        r = int(rng.randint(rmin, rmax + 1))
        specs.append(((x, y), r))
    return specs


def _build_ordinary_source() -> ReferenceFixture:
    rng = np.random.RandomState(101)
    h, w = 900, 1200
    image = _base_canvas((h, w), rng)
    gt = np.zeros((h, w), np.uint8)
    center, radius = (w // 2, h // 2), 355
    _draw_dish(image, center, radius, rng)
    _draw_colonies(image, gt, _scatter(rng, center, radius, 26, 5, 13))
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return ReferenceFixture("ordinary_source", "ordinary", "source", image, gt)


def _build_ordinary_cropped() -> ReferenceFixture:
    rng = np.random.RandomState(102)
    size = 800
    image = _base_canvas((size, size), rng)
    gt = np.zeros((size, size), np.uint8)
    center, radius = (size // 2, size // 2), 392
    _draw_dish(image, center, radius, rng)
    _draw_colonies(image, gt, _scatter(rng, center, radius, 22, 5, 13))
    # Обрезок чашки: чёрный фон вне круга (контракт crop_by_petri).
    outside = np.zeros((size, size), np.uint8)
    cv2.circle(outside, center, radius, 255, -1)
    image[outside == 0] = 0
    gt[outside == 0] = 0
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return ReferenceFixture("ordinary_cropped", "ordinary", "cropped", image, gt)


def _build_boundary_source() -> ReferenceFixture:
    rng = np.random.RandomState(103)
    h, w = 900, 1200
    image = _base_canvas((h, w), rng)
    gt = np.zeros((h, w), np.uint8)
    # Чашка смещена от центра кадра.
    center, radius = (528, 486), 342
    _draw_dish(image, center, radius, rng)
    specs = []
    # Мелкие колонии на границе min_colony_size (площадь ~28–50 px при пороге 50).
    specs += _ring_positions(rng, center, radius, 6, 0.45, 3, 4)
    # Колонии на границе внутреннего ROI (margin 10% => 0.90R).
    specs += _ring_positions(rng, center, radius, 6, 0.90, 5, 8)
    # Колонии у самого края чашки, пересекающие ROI-границу.
    specs += _ring_positions(rng, center, radius, 4, 0.97, 5, 9)
    # Обычные колонии внутри.
    specs += _scatter(rng, center, radius, 8, 5, 13)
    _draw_colonies(image, gt, specs)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return ReferenceFixture("boundary_source", "boundary", "source", image, gt)


def _build_boundary_cropped() -> ReferenceFixture:
    rng = np.random.RandomState(104)
    size = 800
    image = _base_canvas((size, size), rng)
    gt = np.zeros((size, size), np.uint8)
    center, radius = (size // 2, size // 2), 396
    _draw_dish(image, center, radius, rng)
    specs = []
    specs += _ring_positions(rng, center, radius, 5, 0.45, 3, 4)
    specs += _ring_positions(rng, center, radius, 5, 0.88, 5, 8)
    specs += _ring_positions(rng, center, radius, 4, 0.97, 5, 9)
    specs += _scatter(rng, center, radius, 6, 5, 13)
    _draw_colonies(image, gt, specs)
    outside = np.zeros((size, size), np.uint8)
    cv2.circle(outside, center, radius, 255, -1)
    image[outside == 0] = 0
    gt[outside == 0] = 0
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return ReferenceFixture("boundary_cropped", "boundary", "cropped", image, gt)


def _build_largest_source() -> ReferenceFixture:
    rng = np.random.RandomState(105)
    size = 4500  # 20.25 MP — нижняя граница диапазона «крупные 20–35 MP кадры»
    image = _base_canvas((size, size), rng)
    gt = np.zeros((size, size), np.uint8)
    center, radius = (size // 2, size // 2), 1800
    _draw_dish(image, center, radius, rng)
    specs = _scatter(rng, center, radius, 150, 12, 40)
    specs += _ring_positions(rng, center, radius, 10, 0.90, 12, 24)
    _draw_colonies(image, gt, specs)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return ReferenceFixture("largest_source", "largest", "source", image, gt)


def _build_largest_cropped() -> ReferenceFixture:
    rng = np.random.RandomState(106)
    size = 4500
    image = _base_canvas((size, size), rng)
    gt = np.zeros((size, size), np.uint8)
    center, radius = (size // 2, size // 2), 2240
    _draw_dish(image, center, radius, rng)
    specs = _scatter(rng, center, radius, 120, 12, 40)
    specs += _ring_positions(rng, center, radius, 10, 0.95, 12, 24)
    _draw_colonies(image, gt, specs)
    outside = np.zeros((size, size), np.uint8)
    cv2.circle(outside, center, radius, 255, -1)
    image[outside == 0] = 0
    gt[outside == 0] = 0
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return ReferenceFixture("largest_cropped", "largest", "cropped", image, gt)


_BUILDERS = {
    "ordinary_source": _build_ordinary_source,
    "ordinary_cropped": _build_ordinary_cropped,
    "boundary_source": _build_boundary_source,
    "boundary_cropped": _build_boundary_cropped,
    "largest_source": _build_largest_source,
    "largest_cropped": _build_largest_cropped,
}

FIXTURE_NAMES: Tuple[str, ...] = tuple(_BUILDERS)

_cache: Dict[str, ReferenceFixture] = {}


def build_fixture(name: str, use_cache: bool = True) -> ReferenceFixture:
    """Строит эталонную фикстуру; повторная сборка даёт те же байты."""
    if use_cache and name in _cache:
        return _cache[name]
    fixture = _BUILDERS[name]()
    if use_cache:
        _cache[name] = fixture
    return fixture


def build_fixtures(use_cache: bool = True) -> Dict[str, ReferenceFixture]:
    return {name: build_fixture(name, use_cache) for name in FIXTURE_NAMES}


# ----------------------------------------------------------------------------
# Захват baseline: бинарные маски + метрики
# ----------------------------------------------------------------------------


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Приводит маску детекции к каноничному виду uint8 {0, 255}."""
    return ((np.asarray(mask) > 0).astype(np.uint8)) * 255


def array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(f"{array.shape}|{array.dtype}|".encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def mask_sha256(mask: np.ndarray) -> str:
    return array_sha256(normalize_mask(mask))


def mask_file_name(fixture_name: str, algorithm: str) -> str:
    return f"{fixture_name}__{algorithm}.png"


def capture_baseline(
    fixture_names: Iterable[str] | None = None,
    algorithms: Iterable[str] | None = None,
) -> tuple[Dict[str, Dict[str, dict]], Dict[Tuple[str, str], np.ndarray]]:
    """Прогоняет алгоритмы по фикстурам один раз.

    Возвращает (записи с хешами/метриками, каноничные маски по ключам
    `(фикстура, алгоритм)`).
    """
    from testing.metrics import compute_segmentation_metrics
    from testing.registry import get_algorithm

    if fixture_names is None:
        fixture_names = FIXTURE_NAMES
    if algorithms is None:
        algorithms = ALGORITHMS

    entries: Dict[str, Dict[str, dict]] = {}
    masks: Dict[Tuple[str, str], np.ndarray] = {}
    for name in fixture_names:
        fixture = build_fixture(name)
        per_algo: Dict[str, dict] = {}
        for algo_name in algorithms:
            algo = get_algorithm(algo_name)
            pred = algo.detect(fixture.image, is_cropped=fixture.is_cropped)
            mask = normalize_mask(pred)
            masks[(name, algo_name)] = mask
            per_algo[algo_name] = {
                "mask_sha256": mask_sha256(mask),
                "mask_file": f"masks/{mask_file_name(name, algo_name)}",
                "predicted_pixels": int(cv2.countNonZero(mask)),
                "metrics": compute_segmentation_metrics(mask, fixture.ground_truth),
            }
        entries[name] = per_algo
    return entries, masks


def baseline_record(entries: Dict[str, Dict[str, dict]]) -> dict:
    """Полная запись baseline для сохранения в reference_masks.json."""
    fixtures_meta = {}
    for name in entries:
        fixture = build_fixture(name)
        fixtures_meta[name] = {
            "category": fixture.category,
            "variant": fixture.variant,
            "shape": list(fixture.image.shape),
            "image_sha256": array_sha256(fixture.image),
            "ground_truth_sha256": array_sha256(fixture.ground_truth),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "algorithms": list(ALGORITHMS),
        "fixtures": fixtures_meta,
        "entries": entries,
    }


def write_baseline(
    record: dict,
    masks: Dict[Tuple[str, str], np.ndarray],
    json_path: Path | None = None,
) -> None:
    """Пишет JSON и PNG-маски в tests/baseline/."""
    if json_path is None:
        json_path = BASELINE_JSON
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    for (name, algo_name), mask in masks.items():
        out = MASKS_DIR / mask_file_name(name, algo_name)
        cv2.imwrite(str(out), mask)
    json_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_baseline(json_path: Path | None = None) -> dict:
    if json_path is None:
        json_path = BASELINE_JSON
    return json.loads(json_path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# Сравнение с зафиксированным baseline
# ----------------------------------------------------------------------------


def compare_with_baseline(
    entries: Dict[str, Dict[str, dict]],
    masks: Dict[Tuple[str, str], np.ndarray],
    record: dict,
) -> list:
    """Возвращает список расхождений текущего кода с зафиксированным baseline."""
    problems = []
    for name, meta in record.get("fixtures", {}).items():
        fixture = build_fixture(name)
        if array_sha256(fixture.image) != meta["image_sha256"]:
            problems.append(f"{name}: изображение фикстуры изменилось")
        if array_sha256(fixture.ground_truth) != meta["ground_truth_sha256"]:
            problems.append(f"{name}: GT-маска фикстуры изменилась")
    for name, per_algo in record.get("entries", {}).items():
        for algo_name, expected in per_algo.items():
            current = entries.get(name, {}).get(algo_name)
            if current is None:
                problems.append(f"{name}/{algo_name}: нет в текущем прогоне")
                continue
            if current["mask_sha256"] != expected["mask_sha256"]:
                problems.append(
                    f"{name}/{algo_name}: маска изменилась "
                    f"({expected['mask_sha256'][:12]}… -> {current['mask_sha256'][:12]}…)"
                )
            for metric_key, expected_value in expected["metrics"].items():
                current_value = current["metrics"].get(metric_key)
                delta = (
                    abs(float(current_value) - float(expected_value))
                    if current_value is not None
                    else float("inf")
                )
                if delta > 1e-6:
                    problems.append(
                        f"{name}/{algo_name}: метрика {metric_key} "
                        f"{expected_value} -> {current_value} (|Δ|={delta:.3g})"
                    )
            png_path = BASELINE_DIR / expected["mask_file"]
            if not png_path.is_file():
                problems.append(f"{name}/{algo_name}: отсутствует {png_path.name}")
            else:
                stored = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
                current_mask = masks.get((name, algo_name))
                if (
                    stored is None
                    or current_mask is None
                    or not np.array_equal(normalize_mask(stored), current_mask)
                ):
                    problems.append(
                        f"{name}/{algo_name}: PNG-маска {png_path.name} "
                        "не совпадает с текущей побитово"
                    )
    return problems


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Захват/проверка baseline бинарных масок и метрик"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--write", action="store_true", help="перезаписать tests/baseline/"
    )
    group.add_argument(
        "--check", action="store_true", help="сравнить с tests/baseline/"
    )
    args = parser.parse_args(argv)

    entries, masks = capture_baseline()
    if args.write:
        record = baseline_record(entries)
        write_baseline(record, masks)
        print(f"Baseline записан: {BASELINE_JSON}")
        print(f"Маски: {MASKS_DIR} ({len(masks)} файлов)")
        return 0

    record = load_baseline()
    problems = compare_with_baseline(entries, masks, record)
    if problems:
        print("Расхождения с baseline:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("Baseline подтверждён: маски побитово идентичны, метрики в 1e-6.")
    return 0


if __name__ == "__main__":
    # Запуск как скрипта: добавляем корень проекта в sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
