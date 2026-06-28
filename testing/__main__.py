import argparse

from .classic_algorithms import *  # noqa: F401, F403 — triggers @register_algorithm
from .dashboard import generate_report
from .dataset import TestDataset
from .runner import run_all


def main():
    parser = argparse.ArgumentParser(description="Test colony detection algorithms")
    parser.add_argument("--output", default="test_report.html", help="Output HTML report path")
    parser.add_argument("--data-root", default="test_images", help="Test dataset root directory")
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = TestDataset(root=args.data_root)
    print(f"  Found {len(dataset)} samples")

    if len(dataset) == 0:
        print("No test samples found. Populate test_images/ and create GT masks.")
        return

    print("Running algorithms...")
    all_results = run_all(dataset)

    print("Generating report...")
    generate_report(all_results, output_path=args.output)

    print("Done.")


if __name__ == "__main__":
    main()
