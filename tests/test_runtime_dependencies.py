import subprocess
import sys
import tomllib
from pathlib import Path


def test_onnxruntime_is_runtime_dependency_and_pytorch_stays_full_only():
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    runtime = config["project"]["dependencies"]
    full = config["project"]["optional-dependencies"]["full"]

    assert any(dependency.startswith("onnxruntime") for dependency in runtime)
    assert not any(dependency.startswith("onnxruntime") for dependency in full)
    assert any(dependency.startswith("torch") for dependency in full)
    assert not any(dependency.startswith("torch") for dependency in runtime)
    assert any(dependency.startswith("psutil") for dependency in runtime)
    assert config["project"]["scripts"]["train-colony"] == "train.colony_cli:main"


def test_training_cli_module_imports_without_pytorch():
    project_root = Path(__file__).resolve().parents[1]
    script = """
import builtins

original_import = builtins.__import__
def import_without_training_framework(name, *args, **kwargs):
    if name.split(".", 1)[0] in {"torch", "torchvision"}:
        raise ModuleNotFoundError(f"blocked optional training dependency: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_training_framework
import train.colony_cli
import train.evaluation
import train.main
import train.training_pipeline
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
