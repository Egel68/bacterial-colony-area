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
