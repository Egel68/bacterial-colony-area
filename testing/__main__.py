import argparse
import logging

from .classic_algorithms import *  # noqa: F401, F403 — triggers @register_algorithm
from .dashboard import generate_report
from .dataset import TestDataset
from .onnx_algorithm import OnnxModelAlgorithm
from .registry import register_algorithm_instance
from .runner import run_all, compare_algorithms
from utils.logging import setup_logging

log = logging.getLogger(__name__)


def _parse_algorithms(value: str | None) -> list[str] | None:
    """Разбирает список имён алгоритмов через запятую."""
    if value is None:
        return None
    names = [name.strip() for name in value.split(",") if name.strip()]
    return names


def _register_models(model_paths: list[str]) -> None:
    """Регистрирует внешние ONNX-модели в реестре алгоритмов."""
    for path in model_paths:
        algo = OnnxModelAlgorithm(model_path=path)
        register_algorithm_instance(algo.name, algo)
        log.info("Registered model '%s' from %s", algo.name, path)


def main():
    setup_logging(logging.INFO)
    parser = argparse.ArgumentParser(description="Test colony detection algorithms")
    parser.add_argument(
        "--output", default="test_report.html", help="Output HTML report path"
    )
    parser.add_argument(
        "--data-root", default="test_images", help="Test dataset root directory"
    )
    parser.add_argument(
        "--algorithms",
        default=None,
        help="Comma-separated list of algorithms to run (default: all registered)",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Register an external ONNX model file as an algorithm (repeatable)",
    )
    parser.add_argument(
        "--per-snapshot",
        dest="per_snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include per-image tables in the report (default: true)",
    )
    parser.add_argument(
        "--compare",
        default=None,
        metavar="A,B",
        help="Two algorithm names to produce a pairwise comparison section",
    )
    args = parser.parse_args()

    _register_models(args.model)
    algorithms = _parse_algorithms(args.algorithms)

    log.info("Loading dataset...")
    dataset = TestDataset(root=args.data_root)
    log.info("Found %d samples", len(dataset))

    if len(dataset) == 0:
        log.warning("No test samples found. Populate test_images/ and create GT masks.")
        return

    log.info("Running algorithms...")
    all_results = run_all(dataset, algorithms=algorithms)

    comparison = None
    if args.compare:
        a, b = (part.strip() for part in args.compare.split(","))
        log.info("Comparing '%s' vs '%s'", a, b)
        comparison = compare_algorithms(dataset, a, b)

    log.info("Generating report...")
    generate_report(
        all_results,
        output_path=args.output,
        include_per_snapshot=args.per_snapshot,
        comparison=comparison,
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
