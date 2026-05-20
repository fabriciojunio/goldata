"""Testes da CLI do GolData."""

import json
import pytest
from unittest.mock import patch
from io import StringIO

from goldata.cli import cmd_xg, cmd_standings, main


class FakeArgs:
    pass


def test_cli_xg_open_play(capsys):
    args = FakeArgs()
    args.x = 108.0
    args.y = 40.0
    args.penalty = False
    cmd_xg(args)
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert "xg" in result
    assert 0 < result["xg"] < 1


def test_cli_xg_penalty(capsys):
    args = FakeArgs()
    args.x = 108.0
    args.y = 40.0
    args.penalty = True
    cmd_xg(args)
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["xg"] == pytest.approx(0.76)


def test_cli_xg_distance_calculated(capsys):
    args = FakeArgs()
    args.x = 120.0
    args.y = 40.0
    args.penalty = False
    cmd_xg(args)
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["distance"] == pytest.approx(0.0, abs=0.1)


def test_cli_standings_prints_table(capsys):
    args = FakeArgs()
    args.season = 2024
    cmd_standings(args)
    captured = capsys.readouterr()
    assert "Flamengo" in captured.out or "Pts" in captured.out


def test_cli_main_no_command_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        with patch("sys.argv", ["goldata"]):
            main()
    assert exc.value.code == 0


def test_cli_main_xg_command(capsys):
    with patch("sys.argv", ["goldata", "xg", "--x", "110", "--y", "40"]):
        main()
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert "xg" in result


def test_cli_main_standings(capsys):
    with patch("sys.argv", ["goldata", "standings", "--season", "2024"]):
        main()
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_cli_xg_far_shot_low_xg(capsys):
    args = FakeArgs()
    args.x = 50.0
    args.y = 40.0
    args.penalty = False
    cmd_xg(args)
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    # Chute de longe deve ter xG baixo
    assert result["xg"] < 0.2
