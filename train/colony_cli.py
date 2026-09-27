"""Console entrypoint for the reproducible colony-model pipeline."""

import sys


def main() -> None:
    from .main import main as training_main

    training_main(["train-colony", *sys.argv[1:]])
