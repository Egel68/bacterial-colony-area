import argparse
import logging

from .classic_algorithms import *  # noqa: F401, F403 — triggers @register_algorithm
from .dashboard import generate_report
from .dataset import TestDataset
from .runner import run_all
from utils.logging import setup_logging

log = logging.getLogger(__name__)


def main():
    setup_logging(logging.INFO)
    parser = argparse.ArgumentParser(description="Test colony detection algorithms")
    parser.add_argument(
        "--output", default="test_report.html", help="Output HTML report path"
    )
    parser.add_argument(
        "--data-root", default="test_images", help="Test dataset root directory"
    )
    args = parser.parse_args()

    log.info("Loading dataset...")
    dataset = TestDataset(root=args.data_root)
    log.info("Found %d samples", len(dataset))

    if len(dataset) == 0:
        log.warning("No test samples found. Populate test_images/ and create GT masks.")
        return

    log.info("Running algorithms...")
    all_results = run_all(dataset)

    log.info("Generating report...")
    generate_report(all_results, output_path=args.output)

    log.info("Done.")


if __name__ == "__main__":
    main()
