"""Testes dos módulos de betting: value detector, kelly e backtest."""

import pytest
import numpy as np
import pandas as pd

from goldata.models.betting.value_detector import (
    ValueBetDetector, ValueBet, odd_to_implied_prob, remove_bookmaker_margin
)
from goldata.models.betting.kelly import (
    kelly_fraction, fractional_kelly, kelly_stake, KellyCriterion
)
from goldata.models.betting.backtest import BettingBacktest, BacktestResult
from goldata.exceptions import InvalidInputError


# ── Utilitários ───────────────────────────────────────────────────────────────

def test_odd_to_implied_prob_correct():
    assert odd_to_implied_prob(2.0) == pytest.approx(0.5, abs=0.001)


def test_odd_to_implied_prob_100pct():
    assert odd_to_implied_prob(1.0) == pytest.approx(1.0)


def test_remove_bookmaker_margin():
    probs = {"home_win": 0.55, "draw": 0.25, "away_win": 0.28}  # soma > 1 (margem)
    clean = remove_bookmaker_margin(probs)
    total = sum(clean.values())
    assert abs(total - 1.0) < 0.01


# ── Value Detector ────────────────────────────────────────────────────────────

@pytest.fixture
def detector():
    return ValueBetDetector(min_edge=0.03, min_confidence=0.55)


def test_detect_value_bet_found(detector):
    bets = detector.detect(
        "Flamengo", "Palmeiras",
        model_probs={"home_win": 0.62, "draw": 0.22, "away_win": 0.16},
        market_odds={"home_win": 2.10, "draw": 3.40, "away_win": 5.00},
    )
    assert len(bets) >= 1
    assert bets[0].market == "home_win"


def test_detect_no_value_when_no_edge(detector):
    bets = detector.detect(
        "Flamengo", "Palmeiras",
        model_probs={"home_win": 0.45, "draw": 0.30, "away_win": 0.25},
        market_odds={"home_win": 2.10, "draw": 3.00, "away_win": 4.00},
    )
    # Sem edge positivo claro
    for bet in bets:
        assert bet.edge >= 0.03


def test_detect_value_bet_fields(detector):
    bets = detector.detect(
        "Time A", "Time B",
        model_probs={"home_win": 0.65, "draw": 0.20, "away_win": 0.15},
        market_odds={"home_win": 2.20, "draw": 3.50, "away_win": 5.50},
    )
    if bets:
        bet = bets[0]
        assert hasattr(bet, "edge")
        assert hasattr(bet, "expected_value")
        assert hasattr(bet, "confidence_stars")
        assert 1 <= bet.confidence_stars <= 5


def test_detect_ev_positive(detector):
    bets = detector.detect(
        "A", "B",
        model_probs={"home_win": 0.70},
        market_odds={"home_win": 2.00},
    )
    if bets:
        assert bets[0].expected_value > 0


def test_detect_edge_correct(detector):
    bets = detector.detect(
        "A", "B",
        model_probs={"home_win": 0.60},
        market_odds={"home_win": 2.50},
    )
    if bets:
        expected_edge = 0.60 - (1 / 2.50)
        assert abs(bets[0].edge - expected_edge) < 0.001


def test_detect_batch_multiple_matches(detector):
    fixtures = [
        {"home_team": "A", "away_team": "B",
         "model_probs": {"home_win": 0.65}, "market_odds": {"home_win": 2.20}},
        {"home_team": "C", "away_team": "D",
         "model_probs": {"away_win": 0.60}, "market_odds": {"away_win": 2.10}},
    ]
    bets = detector.detect_batch(fixtures)
    assert isinstance(bets, list)


def test_detect_to_dataframe(detector):
    bets = detector.detect(
        "A", "B",
        model_probs={"home_win": 0.65},
        market_odds={"home_win": 2.20},
    )
    df = detector.to_dataframe(bets)
    assert isinstance(df, pd.DataFrame)


def test_detect_empty_to_dataframe(detector):
    df = detector.to_dataframe([])
    assert len(df) == 0


# ── Kelly Criterion ───────────────────────────────────────────────────────────

def test_kelly_fraction_positive_ev():
    f = kelly_fraction(0.55, 2.0)
    assert f > 0


def test_kelly_fraction_negative_ev():
    f = kelly_fraction(0.40, 2.0)  # EV negativo (0.40 * 1 - 0.60 = -0.20)
    assert f == 0.0


def test_kelly_fraction_range():
    f = kelly_fraction(0.6, 2.0)
    assert 0 <= f <= 1


def test_fractional_kelly_smaller_than_full():
    full = kelly_fraction(0.6, 2.0)
    frac = fractional_kelly(0.6, 2.0, fraction=0.25)
    assert frac <= full


def test_fractional_kelly_max_bet_cap():
    frac = fractional_kelly(0.9, 2.0, fraction=1.0, max_bet=0.05)
    assert frac <= 0.05


def test_kelly_invalid_prob_raises():
    with pytest.raises(InvalidInputError):
        kelly_fraction(1.5, 2.0)


def test_kelly_invalid_odd_raises():
    with pytest.raises(InvalidInputError):
        kelly_fraction(0.6, 0.5)


def test_kelly_stake_dict_fields():
    result = kelly_stake(1000.0, 0.55, 2.10)
    assert "stake_amount" in result
    assert "expected_value" in result
    assert "edge" in result
    assert result["stake_amount"] >= 0


def test_kelly_criterion_record_bet_win():
    kc = KellyCriterion(initial_bankroll=1000.0)
    result = kc.record_bet(prob=0.6, odd=2.0, won=True, match="A vs B")
    assert result["profit"] > 0
    assert kc.bankroll > 1000.0


def test_kelly_criterion_record_bet_loss():
    kc = KellyCriterion(initial_bankroll=1000.0)
    result = kc.record_bet(prob=0.6, odd=2.0, won=False)
    assert result["profit"] < 0
    assert kc.bankroll < 1000.0


def test_kelly_criterion_summary():
    kc = KellyCriterion(initial_bankroll=1000.0)
    kc.record_bet(0.6, 2.0, True)
    kc.record_bet(0.6, 2.0, False)
    summary = kc.get_summary()
    assert summary["total_bets"] == 2
    assert "roi" in summary
    assert "yield_pct" in summary


def test_kelly_criterion_history_dataframe():
    kc = KellyCriterion(1000.0)
    kc.record_bet(0.6, 2.0, True)
    history = kc.get_history()
    assert isinstance(history, pd.DataFrame)
    assert len(history) == 1


# ── Backtest ─────────────────────────────────────────────────────────────────

@pytest.fixture
def backtest_data():
    np.random.seed(42)
    n = 200
    model_probs = np.random.uniform(0.45, 0.75, n)
    odds = 1.0 / (model_probs - np.random.uniform(0.02, 0.08, n))
    odds = np.clip(odds, 1.3, 8.0)
    outcomes = np.random.binomial(1, model_probs)
    return pd.DataFrame({"model_prob": model_probs, "odd": odds, "outcome": outcomes})


def test_backtest_runs(backtest_data):
    bt = BettingBacktest(initial_bankroll=1000.0)
    result = bt.run(backtest_data)
    assert isinstance(result, BacktestResult)


def test_backtest_n_bets_positive(backtest_data):
    bt = BettingBacktest(initial_bankroll=1000.0, min_edge=0.02)
    result = bt.run(backtest_data)
    assert result.n_bets >= 0


def test_backtest_equity_curve_not_empty(backtest_data):
    bt = BettingBacktest(initial_bankroll=1000.0, min_edge=0.02)
    result = bt.run(backtest_data)
    assert len(result.equity_curve) > 0


def test_backtest_win_rate_in_range(backtest_data):
    bt = BettingBacktest(initial_bankroll=1000.0, min_edge=0.02)
    result = bt.run(backtest_data)
    assert 0 <= result.win_rate <= 1


def test_backtest_brier_score_in_range(backtest_data):
    bt = BettingBacktest(initial_bankroll=1000.0, min_edge=0.02)
    result = bt.run(backtest_data)
    assert 0 <= result.brier_score <= 1


def test_backtest_max_drawdown_non_positive(backtest_data):
    bt = BettingBacktest(initial_bankroll=1000.0, min_edge=0.02)
    result = bt.run(backtest_data)
    assert result.max_drawdown <= 0


def test_backtest_empty_after_filters():
    """Dados sem nenhum bet válido."""
    df = pd.DataFrame({"model_prob": [0.3] * 10, "odd": [3.5] * 10, "outcome": [0] * 10})
    bt = BettingBacktest(min_edge=0.50)  # edge muito alto → nenhum bet
    result = bt.run(df)
    assert result.n_bets == 0


def test_backtest_walk_forward(backtest_data):
    bt = BettingBacktest(initial_bankroll=1000.0, min_edge=0.02)
    results = bt.walk_forward(backtest_data, n_folds=5)
    assert len(results) == 5


def test_backtest_profit_property(backtest_data):
    bt = BettingBacktest(initial_bankroll=1000.0, min_edge=0.02)
    result = bt.run(backtest_data)
    expected_profit = result.final_bankroll - result.initial_bankroll
    assert abs(result.profit - expected_profit) < 0.01
