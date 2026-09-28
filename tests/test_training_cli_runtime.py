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
