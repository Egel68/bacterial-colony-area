import os
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
    # Папка с исходными наборами данных может быть namespace package для Nuitka,
    # но не является Python-кодом приложения; не проверять её метаданные пакета.
    flags.append("--nofollow-import-to=datasets")

    # ONNX Runtime содержит пакет python-модулей, CPython extension и runtime
    # shared libraries; все они нужны bundled-моделям в inference build.
    flags.extend(
        [
            "--include-package=onnxruntime",
            "--include-package-data=onnxruntime",
        ]
    )
    build_env = Path(os.environ.get("UV_PROJECT_ENVIRONMENT", ".venv-build"))
    if not build_env.is_absolute():
        build_env = root / build_env
    site_packages = list(build_env.glob("lib/python*/site-packages"))
    windows_site_packages = build_env / "Lib" / "site-packages"
    if windows_site_packages.is_dir():
        site_packages.append(windows_site_packages)
    ort_capi_dirs = [
        path / "onnxruntime" / "capi"
        for path in site_packages
        if (path / "onnxruntime" / "capi").is_dir()
    ]
    for ort_capi in ort_capi_dirs:
        # Nuitka's onnxruntime package config collects the python extensions and
        # provider helpers. Explicitly include only the dynamically loaded ORT
        # runtime library to avoid duplicate destinations for provider libraries.
        for pattern in (
            "libonnxruntime.so",
            "libonnxruntime.so.*",
            "libonnxruntime.dylib",
            "onnxruntime.dll",
        ):
            for library in sorted(ort_capi.glob(pattern)):
                flags.append(
                    f"--include-data-files={library}=onnxruntime/capi/{library.name}"
                )

    # Встроенные ONNX-модели: кладём весь каталог models/*.onnx в бинарник.
    models_dir = root / "models"
    if models_dir.is_dir():
        onnx_files = sorted(models_dir.glob("*.onnx"))
        if onnx_files:
            flags.append(f"--include-data-files={models_dir.resolve()}/*.onnx=models/")

    return flags


if __name__ == "__main__":
    print(" ".join(build_flags(ROOT)))
