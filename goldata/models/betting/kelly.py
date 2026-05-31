"""Kelly Criterion para dimensionamento ótimo de apostas."""

import numpy as np
import pandas as pd

from goldata.exceptions import InvalidInputError
from goldata.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_KELLY_FRACTION = 0.25  # 1/4 Kelly: mais conservador
MAX_BET_FRACTION = 0.05        # máximo 5% do bankroll por aposta


def kelly_fraction(prob: float, odd: float) -> float:
    """
    Calcula a fração Kelly ótima.

    f* = (b*p - q) / b
    onde b = odd - 1, p = probabilidade de vitória, q = 1 - p

    Args:
        prob: probabilidade estimada pelo modelo [0, 1]
        odd: odd decimal da casa de apostas

    Returns:
        Fração ótima do bankroll a apostar [0, 1]
    """
    if not (0 < prob < 1):
        raise InvalidInputError("prob", prob, "deve estar entre 0 e 1 exclusive")
    if odd < 1.0:
        raise InvalidInputError("odd", odd, "odd deve ser >= 1.0")

    b = odd - 1.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    return max(0.0, round(f, 6))


def fractional_kelly(
    prob: float,
    odd: float,
    fraction: float = DEFAULT_KELLY_FRACTION,
    max_bet: float = MAX_BET_FRACTION,
) -> float:
    """
    Kelly fracionado (mais conservador e prático).

    Args:
        prob: probabilidade estimada
        odd: odd decimal
        fraction: fração do Kelly completo a usar (default 0.25 = 1/4 Kelly)
        max_bet: teto máximo como fração do bankroll

    Returns:
        Fração do bankroll a apostar
    """
    full_kelly = kelly_fraction(prob, odd)
    adjusted = full_kelly * fraction
    return min(adjusted, max_bet)


def kelly_stake(
    bankroll: float,
    prob: float,
    odd: float,
    fraction: float = DEFAULT_KELLY_FRACTION,
    max_bet: float = MAX_BET_FRACTION,
) -> dict:
    """
    Calcula o valor absoluto da aposta usando Kelly fracionado.

    Args:
        bankroll: saldo atual em reais
        prob: probabilidade estimada
        odd: odd decimal

    Returns:
        dict com bet_fraction, stake_amount, expected_profit, edge
    """
    bet_fraction = fractional_kelly(prob, odd, fraction, max_bet)
    stake = round(bankroll * bet_fraction, 2)
    edge = round(prob - (1.0 / odd), 4)
    ev = round(odd * prob - 1.0, 4)
    expected_profit = round(stake * ev, 2)

    return {
        "bet_fraction": round(bet_fraction, 6),
        "stake_amount": stake,
        "edge": edge,
        "expected_value": ev,
        "expected_profit": expected_profit,
        "break_even_prob": round(1.0 / odd, 4),
    }


class KellyCriterion:
    """
    Gerenciador de banca usando Kelly Criterion.

    Mantém histórico de apostas e calcula sizing ótimo
    para cada oportunidade detectada.
    """

    def __init__(
        self,
        initial_bankroll: float,
        kelly_fraction: float = DEFAULT_KELLY_FRACTION,
        max_bet_fraction: float = MAX_BET_FRACTION,
    ) -> None:
        self.bankroll = initial_bankroll
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        self._bet_history: list[dict] = []

    def calculate_stake(self, prob: float, odd: float) -> dict:
        """Calcula stake para a banca atual."""
        return kelly_stake(
            self.bankroll, prob, odd,
            self.kelly_fraction, self.max_bet_fraction,
        )

    def record_bet(
        self,
        prob: float,
        odd: float,
        won: bool,
        match: str = "",
        market: str = "",
    ) -> dict:
        """
        Registra o resultado de uma aposta e atualiza a banca.

        Returns:
            dict com resultado e nova banca
        """
        stake_info = self.calculate_stake(prob, odd)
        stake = stake_info["stake_amount"]

        if won:
            profit = round(stake * (odd - 1.0), 2)
        else:
            profit = -stake

        self.bankroll = round(self.bankroll + profit, 2)

        record = {
            "match": match,
            "market": market,
            "prob": prob,
            "odd": odd,
            "stake": stake,
            "won": won,
            "profit": profit,
            "bankroll_after": self.bankroll,
            "roi_cumulative": round((self.bankroll - self.initial_bankroll) / self.initial_bankroll, 4),
        }
        self._bet_history.append(record)
        return record

    def get_history(self) -> pd.DataFrame:
        """Retorna histórico de apostas como DataFrame."""
        if not self._bet_history:
            return pd.DataFrame()
        return pd.DataFrame(self._bet_history)

    def get_summary(self) -> dict:
        """Retorna resumo da banca."""
        if not self._bet_history:
            return {
                "total_bets": 0, "wins": 0, "losses": 0,
                "current_bankroll": self.bankroll,
                "roi": 0.0, "yield_pct": 0.0,
            }
        history = self.get_history()
        total_staked = history["stake"].sum()
        total_profit = history["profit"].sum()
        return {
            "total_bets": len(history),
            "wins": int(history["won"].sum()),
            "losses": int((~history["won"]).sum()),
            "win_rate": round(float(history["won"].mean()), 4),
            "current_bankroll": self.bankroll,
            "total_staked": round(float(total_staked), 2),
            "total_profit": round(float(total_profit), 2),
            "roi": round(float((self.bankroll - self.initial_bankroll) / self.initial_bankroll), 4),
            "yield_pct": round(float(total_profit / total_staked * 100) if total_staked > 0 else 0.0, 2),
        }
