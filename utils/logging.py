import logging


def setup_logging(level: int = logging.WARNING) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
        datefmt="%H:%M:%S",
    )
