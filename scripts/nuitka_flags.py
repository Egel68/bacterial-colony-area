from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXCLUDE = {
    ".venv",
    ".venv-dev",
    ".venv-full",
    ".venv-build",
    "__pycache__",
    "bacterial_colony_analyzer.egg-info",
    ".git",
    ".github",
    "scripts",
    "test_images",
    "test_data",
    "train",
}


def build_flags(root: Path) -> list[str]:
    """Формирует флаги --include-package и --enable-plugin=pyqt6 (единый источник)."""
    packages = sorted(
        e.name
        for e in root.iterdir()
        if e.is_dir()
        and e.name not in EXCLUDE
        and not e.name.startswith(".")
        and (e / "__init__.py").exists()
    )
    flags = [f"--include-package={p}" for p in packages]
    flags.append("--enable-plugin=pyqt6")

    # Встроенные ONNX-модели: кладём весь каталог models/*.onnx в бинарник.
    models_dir = root / "models"
    if models_dir.is_dir():
        onnx_files = sorted(models_dir.glob("*.onnx"))
        if onnx_files:
            flags.append("--include-data-files=models/*.onnx=models/")

    return flags


if __name__ == "__main__":
    print(" ".join(build_flags(ROOT)))
