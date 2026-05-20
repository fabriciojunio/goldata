"""Backtesting de estratégias de apostas com walk-forward validation."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from goldata.models.betting.kelly import KellyCriterion, fractional_kelly
from goldata.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Resultado de um backtest."""
    n_bets: int
    n_wins: int
    roi: float
    yield_pct: float
    max_drawdown: float
    sharpe_ratio: float
    brier_score: float
    final_bankroll: float
    initial_bankroll: float
    equity_curve: list[float] = field(default_factory=list)
    monthly_returns: dict[str, float] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_bets if self.n_bets > 0 else 0.0

    @property
    def profit(self) -> float:
        return self.final_bankroll - self.initial_bankroll


class BettingBacktest:
    """
    Backtesta estratégias de apostas com dados históricos.

    Usa walk-forward validation: treinar em N temporadas, testar na N+1.
    Calcula métricas standard do betting: ROI, Yield, Sharpe, Max Drawdown.
    """

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.03,
        min_odds: float = 1.30,
        max_odds: float = 10.0,
    ) -> None:
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.min_odds = min_odds
        self.max_odds = max_odds

    def run(
        self,
        predictions_df: pd.DataFrame,
    ) -> BacktestResult:
        """
        Executa o backtest em dados históricos de previsões.

        Args:
            predictions_df: DataFrame com colunas:
                - model_prob: probabilidade estimada pelo modelo
                - odd: odd da casa de apostas
                - outcome: 1 se ganhou, 0 se perdeu

        Returns:
            BacktestResult com todas as métricas
        """
        bankroll = self.initial_bankroll
        equity_curve = [bankroll]
        bet_profits = []

        required = ["model_prob", "odd", "outcome"]
        for col in required:
            if col not in predictions_df.columns:
                raise ValueError(f"Coluna obrigatória ausente: {col}")

        # Filtrar apostas válidas
        df = predictions_df[
            (predictions_df["odd"] >= self.min_odds)
            & (predictions_df["odd"] <= self.max_odds)
        ].copy()

        df["implied_prob"] = 1.0 / df["odd"]
        df["edge"] = df["model_prob"] - df["implied_prob"]
        df = df[df["edge"] >= self.min_edge].copy()

        n_wins = 0
        for _, row in df.iterrows():
            stake_frac = fractional_kelly(
                float(row["model_prob"]),
                float(row["odd"]),
                self.kelly_fraction,
            )
            stake = bankroll * stake_frac

            won = bool(row["outcome"])
            if won:
                profit = stake * (float(row["odd"]) - 1.0)
                n_wins += 1
            else:
                profit = -stake

            bankroll += profit
            bankroll = max(bankroll, 0.01)  # não ir abaixo de 0
            equity_curve.append(bankroll)
            bet_profits.append(profit)

        # Métricas
        n_bets = len(df)
        if n_bets == 0:
            return BacktestResult(
                n_bets=0, n_wins=0, roi=0.0, yield_pct=0.0,
                max_drawdown=0.0, sharpe_ratio=0.0, brier_score=0.0,
                final_bankroll=self.initial_bankroll,
                initial_bankroll=self.initial_bankroll,
                equity_curve=equity_curve,
            )

        profits_arr = np.array(bet_profits)
        stakes_arr = np.array([
            self.initial_bankroll * fractional_kelly(
                float(r["model_prob"]), float(r["odd"]), self.kelly_fraction
            )
            for _, r in df.iterrows()
        ])

        roi = float((bankroll - self.initial_bankroll) / self.initial_bankroll)
        total_staked = float(stakes_arr.sum())
        yield_pct = float(profits_arr.sum() / total_staked * 100) if total_staked > 0 else 0.0

        # Max drawdown
        equity_arr = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - running_max) / running_max
        max_drawdown = float(drawdowns.min())

        # Sharpe (simplificado, usando retornos por aposta)
        returns = profits_arr / stakes_arr if len(stakes_arr) > 0 else np.array([0.0])
        sharpe = float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0

        # Brier Score (calibração)
        brier = float(np.mean((df["model_prob"].values - df["outcome"].values) ** 2))

        logger.info(
            "backtest_completed",
            n_bets=n_bets, roi=round(roi, 4), yield_pct=round(yield_pct, 2),
        )

        return BacktestResult(
            n_bets=n_bets,
            n_wins=n_wins,
            roi=round(roi, 4),
            yield_pct=round(yield_pct, 2),
            max_drawdown=round(max_drawdown, 4),
            sharpe_ratio=round(sharpe, 4),
            brier_score=round(brier, 4),
            final_bankroll=round(bankroll, 2),
            initial_bankroll=self.initial_bankroll,
            equity_curve=equity_curve,
        )

    def walk_forward(
        self,
        predictions_df: pd.DataFrame,
        n_folds: int = 5,
    ) -> list[BacktestResult]:
        """
        Walk-forward validation: testa em folds temporais.

        Args:
            predictions_df: DataFrame com previsões ordenadas por data
            n_folds: número de folds

        Returns:
            Lista de BacktestResult, um por fold
        """
        n = len(predictions_df)
        fold_size = n // n_folds
        results = []

        for i in range(1, n_folds + 1):
            test_end = i * fold_size
            test_start = test_end - fold_size
            test_df = predictions_df.iloc[test_start:test_end]
            result = self.run(test_df)
            results.append(result)

        return results
