"""Проверки сохранения CLI-инструментов внутренней валидации."""

import pytest


def test_test_algorithms_cli_retains_selection_and_external_model_options(
    monkeypatch, capsys
):
    from testing.__main__ import main

    monkeypatch.setattr("sys.argv", ["test-algorithms", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--algorithms" in output
    assert "--model" in output
    assert "--compare" in output
