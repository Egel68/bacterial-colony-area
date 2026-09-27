"""Console entrypoint for the reproducible colony-model pipeline."""

import sys

from .main import main as training_main


def main() -> None:
    training_main(["train-colony", *sys.argv[1:]])
