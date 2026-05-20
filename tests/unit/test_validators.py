"""Testes dos validadores de dados."""

import pytest
from goldata.data.validators import DataValidator
from goldata.exceptions import InvalidInputError


v = DataValidator()


def test_valid_shot_data():
    assert v.validate_shot_data({"x": 105, "y": 40, "xg": 0.15}) is True


def test_shot_x_out_of_range():
    with pytest.raises(InvalidInputError):
        v.validate_shot_data({"x": 130})


def test_shot_y_out_of_range():
    with pytest.raises(InvalidInputError):
        v.validate_shot_data({"y": 90})


def test_shot_xg_above_1():
    with pytest.raises(InvalidInputError):
        v.validate_shot_data({"xg": 1.5})


def test_shot_xg_below_0():
    with pytest.raises(InvalidInputError):
        v.validate_shot_data({"xg": -0.1})


def test_valid_match_result():
    assert v.validate_match_result(2, 1) is True


def test_valid_draw():
    assert v.validate_match_result(0, 0) is True


def test_negative_home_score():
    with pytest.raises(InvalidInputError):
        v.validate_match_result(-1, 0)


def test_negative_away_score():
    with pytest.raises(InvalidInputError):
        v.validate_match_result(0, -1)


def test_unrealistic_score():
    with pytest.raises(InvalidInputError):
        v.validate_match_result(25, 0)


def test_valid_player_stats():
    assert v.validate_player_stats({"goals_per_90": 0.5, "assists_per_90": 0.3}) is True


def test_negative_per90_stat():
    with pytest.raises(InvalidInputError):
        v.validate_player_stats({"goals_per_90": -1})


def test_unrealistic_per90_stat():
    with pytest.raises(InvalidInputError):
        v.validate_player_stats({"goals_per_90": 15})


def test_valid_odds():
    assert v.validate_odds(2.5) is True


def test_odds_below_1():
    with pytest.raises(InvalidInputError):
        v.validate_odds(0.5)


def test_valid_probability():
    assert v.validate_probability(0.7) is True


def test_probability_above_1():
    with pytest.raises(InvalidInputError):
        v.validate_probability(1.5)
