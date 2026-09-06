import argparse
import logging

from .baseline import BaselineDataset, run_baseline, export_json
from .classic_algorithms import *  # noqa: F401, F403 — triggers @register_algorithm
from .evaluator import run_evaluate, load_config, EVALUATIONS_DIR
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
        "--mode",
        default="report",
        choices=["report", "baseline", "evaluate"],
        help="Mode: 'report' (default, HTML report), 'baseline' (JSON baseline metrics), or 'evaluate' (full pipeline with run dir)",
    )
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
    parser.add_argument(
        "--no-cropped",
        dest="use_cropped",
        action="store_false",
        default=True,
        help="Disable cropped variant processing in baseline mode",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML/JSON config file (evaluate mode)",
    )
    parser.add_argument(
        "--import-root",
        default=None,
        help="Import dataset from this root before evaluation (evaluate mode)",
    )
    args = parser.parse_args()

    if args.mode == "evaluate":
        return _run_evaluate(args)
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


def _run_evaluate(args):
    log.info("Evaluate mode")
    cli_dict = {
        "data_root": args.data_root,
        "use_cropped": args.use_cropped,
        "import_root": args.import_root,
    }
    config = load_config(args.config, cli_dict)
    if args.import_root is not None:
        config["import_root"] = args.import_root
    run_evaluate(config)
    log.info("Evaluation complete")


def _run_baseline(args):
    log.info("Baseline mode: %s", args.data_root)
    dataset = BaselineDataset(root=args.data_root)
    log.info(
        "Dataset: %d source, %d cropped samples",
        dataset.count_source(),
        dataset.count_cropped(),
    )
    if len(dataset) == 0:
        log.warning("No test samples found.")
        return
    result = run_baseline(dataset, use_cropped=args.use_cropped)
    export_json(result, args.output)
    log.info("Baseline report written to %s", args.output)


def main_evaluate():
    """Entry point for `uv run evaluate`. Defaults to evaluate mode."""
    setup_logging(logging.INFO)
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline")
    parser.add_argument("--data-root", default="test_images")
    parser.add_argument("--no-cropped", dest="use_cropped", action="store_false", default=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--import-root", default=None)
    args = parser.parse_args()
    log.info("Evaluate mode")
    cli_dict = {
        "data_root": args.data_root,
        "use_cropped": args.use_cropped,
        "import_root": args.import_root,
    }
    config = load_config(args.config, cli_dict)
    if args.import_root is not None:
        config["import_root"] = args.import_root
    run_evaluate(config)
    log.info("Evaluation complete")


if __name__ == "__main__":
    main()
