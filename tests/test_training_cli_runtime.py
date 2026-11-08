import pytest

from train.main import main


def test_training_cli_help_lists_reproducible_options(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["train-colony", "--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    for option in (
        "--data-root",
        "--output",
        "--model",
        "--black-background",
        "--reuse-prepared-dataset",
        "--seed",
        "--train-ratio",
        "--val-ratio",
        "--test-ratio",
        "--patches-per-image",
        "--no-pretrained",
        "--promote-model",
        "--resume-run",
    ):
        assert option in output


def test_train_colony_preserves_default_model_and_wires_experiment_flags(
    monkeypatch,
):
    import train.training_pipeline as pipeline

    calls = []
    monkeypatch.setattr(
        pipeline, "run_colony_training", lambda **kwargs: calls.append(kwargs)
    )

    main(["train-colony", "--data-root", "input", "--output", "out"])
    assert calls[-1]["model_name"] == "mobilenet_v3_small_unet"
    assert calls[-1]["black_background"] is False
    assert calls[-1]["reuse_prepared_dataset"] is False

    main(
        [
            "train-colony",
            "--data-root",
            "input",
            "--output",
            "out",
            "--model",
            "unet",
            "--black-background",
        ]
    )
    assert calls[-1]["model_name"] == "unet"
    assert calls[-1]["black_background"] is True
    assert calls[-1]["reuse_prepared_dataset"] is False

    main(
        [
            "train-colony",
            "--data-root",
            "input",
            "--output",
            "out",
            "--model",
            "unet",
            "--black-background",
            "--reuse-prepared-dataset",
        ]
    )
    assert calls[-1]["reuse_prepared_dataset"] is True


def test_black_background_cli_requires_fresh_scratch_unet(capsys):
    base = ["train-colony", "--data-root", "input", "--output", "out"]
    with pytest.raises(SystemExit) as exc:
        main([*base, "--black-background"])
    assert exc.value.code == 2
    assert "requires --model unet" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        main(
            [*base, "--model", "unet", "--black-background", "--resume-run", "old-run"]
        )
    assert exc.value.code == 2
    assert "fresh run" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        main([*base, "--model", "unet", "--reuse-prepared-dataset"])
    assert exc.value.code == 2
    assert "requires --black-background" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        main(
            [
                *base,
                "--model",
                "unet",
                "--black-background",
                "--reuse-prepared-dataset",
                "--resume-run",
                "old-run",
            ]
        )
    assert exc.value.code == 2
    assert "incompatible with --resume-run" in capsys.readouterr().err


def test_training_cli_rejects_invalid_split_ratio(capsys):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "train-colony",
                "--data-root",
                "dataset",
                "--output",
                "output",
                "--train-ratio",
                "0.7",
                "--val-ratio",
                "0.2",
                "--test-ratio",
                "0.2",
            ]
        )

    assert exc.value.code == 2
    assert "sum to 1" in capsys.readouterr().err


def test_training_refuses_nonempty_output_without_touching_existing_data(tmp_path):
    from train.training_pipeline import run_colony_training

    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "user-data.txt"
    marker.write_text("preserve")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        run_colony_training(data_root=source, output_root=output, pretrained=False)

    assert marker.read_text() == "preserve"


def test_incomplete_prepared_pair_is_rejected_before_evaluation(tmp_path):
    from types import SimpleNamespace

    from train.training_pipeline import _validate_black_background_dataset

    source = SimpleNamespace(id="dish-1", kind="source")
    manifest = SimpleNamespace(samples=[source])
    with pytest.raises(ValueError, match="exactly one black-background variant"):
        _validate_black_background_dataset(manifest, tmp_path, {"dish-1"})


def test_black_background_pipeline_requires_cuda_before_import(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from train.training_pipeline import run_colony_training

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        run_colony_training(
            data_root=tmp_path / "missing-source",
            output_root=tmp_path / "output",
            model_name="unet",
            black_background=True,
        )


def test_training_loop_requires_cuda_without_cpu_fallback(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from train.config import TrainingConfig
    from train.train import run_training

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = TrainingConfig(data_root=tmp_path / "missing-dataset", require_cuda=True)
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        run_training(cfg)
