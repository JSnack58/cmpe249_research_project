import pytest


def test_cli_lists_stages(capsys):
    from plate_cloak import cli

    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for stage in ["prepare-data", "finetune", "attack", "evaluate", "run", "report"]:
        assert stage in out
