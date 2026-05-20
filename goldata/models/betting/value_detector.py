"""Detecção de value bets comparando probabilidades do modelo com odds de mercado."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from goldata.data.validators import DataValidator
from goldata.logging_config import get_logger

logger = get_logger(__name__)

validator = DataValidator()

MINIMUM_EDGE = 0.03  # 3% de edge mínimo para considerar value
MINIMUM_CONFIDENCE = 0.55  # 55% de probabilidade mínima


@dataclass
class ValueBet:
    """Representa uma aposta com valor positivo detectado."""
    home_team: str
    away_team: str
    market: str             # "home_win", "draw", "away_win"
    model_prob: float       # Probabilidade estimada pelo modelo
    implied_prob: float     # Probabilidade implícita da odd
    odd: float              # Odd da casa de apostas
    edge: float             # Vantagem esperada (model_prob - implied_prob)
    expected_value: float   # EV = odd * model_prob - 1
    confidence_stars: int   # 1-5 estrelas de confiança


def odd_to_implied_prob(odd: float) -> float:
    """Converte odd decimal em probabilidade implícita (sem margem removida)."""
    validator.validate_odds(odd)
    return round(1.0 / odd, 4)


def remove_bookmaker_margin(
    probs: dict[str, float],
    outcomes: list[str] | None = None,
) -> dict[str, float]:
    """
    Remove a margem da casa de apostas das probabilidades implícitas.

    Args:
        probs: dict {market: implied_prob}
        outcomes: lista de mercados (usa chaves do dict se None)

    Returns:
        dict com probabilidades normalizadas (sem margem)
    """
    keys = outcomes or list(probs.keys())
    total = sum(probs[k] for k in keys if k in probs)
    if total == 0:
        return probs
    return {k: round(probs[k] / total, 4) for k in keys if k in probs}


class ValueBetDetector:
    """
    Detecta apostas com valor positivo comparando o modelo com o mercado.

    Edge = modelo_prob - implied_prob
    EV = odd * modelo_prob - 1

    Uma aposta tem "value" quando:
    - Edge > MINIMUM_EDGE (3%)
    - EV > 0
    - Confiança do modelo > MINIMUM_CONFIDENCE
    """

    def __init__(
        self,
        min_edge: float = MINIMUM_EDGE,
        min_confidence: float = MINIMUM_CONFIDENCE,
    ) -> None:
        self.min_edge = min_edge
        self.min_confidence = min_confidence

    def _confidence_stars(self, edge: float, model_prob: float) -> int:
        """Converte edge e confiança em 1-5 estrelas."""
        score = edge * 10 + max(0, model_prob - 0.5) * 5
        if score >= 1.5:
            return 5
        elif score >= 1.0:
            return 4
        elif score >= 0.6:
            return 3
        elif score >= 0.3:
            return 2
        return 1

    def detect(
        self,
        home_team: str,
        away_team: str,
        model_probs: dict[str, float],
        market_odds: dict[str, float],
    ) -> list[ValueBet]:
        """
        Detecta value bets para uma partida.

        Args:
            home_team: nome do time da casa
            away_team: nome do time visitante
            model_probs: {"home_win": 0.55, "draw": 0.25, "away_win": 0.20}
            market_odds: {"home_win": 1.80, "draw": 3.40, "away_win": 4.50}

        Returns:
            Lista de ValueBet com edge positivo
        """
        markets = ["home_win", "draw", "away_win"]
        value_bets = []

        for market in markets:
            if market not in model_probs or market not in market_odds:
                continue

            odd = float(market_odds[market])
            model_prob = float(model_probs[market])
            implied_prob = odd_to_implied_prob(odd)
            edge = round(model_prob - implied_prob, 4)
            ev = round(odd * model_prob - 1.0, 4)

            if edge >= self.min_edge and model_prob >= self.min_confidence and ev > 0:
                value_bets.append(ValueBet(
                    home_team=home_team,
                    away_team=away_team,
                    market=market,
                    model_prob=round(model_prob, 4),
                    implied_prob=round(implied_prob, 4),
                    odd=odd,
                    edge=edge,
                    expected_value=ev,
                    confidence_stars=self._confidence_stars(edge, model_prob),
                ))

        logger.info(
            "value_bets_detected",
            match=f"{home_team} vs {away_team}",
            n_value_bets=len(value_bets),
        )
        return value_bets

    def detect_batch(
        self,
        fixtures: list[dict[str, Any]],
    ) -> list[ValueBet]:
        """
        Detecta value bets para múltiplas partidas.

        Args:
            fixtures: lista de dicts com home_team, away_team, model_probs, market_odds

        Returns:
            Lista consolidada de value bets, ordenada por EV decrescente
        """
        all_bets = []
        for fixture in fixtures:
            bets = self.detect(
                home_team=fixture["home_team"],
                away_team=fixture["away_team"],
                model_probs=fixture["model_probs"],
                market_odds=fixture["market_odds"],
            )
            all_bets.extend(bets)

        all_bets.sort(key=lambda x: -x.expected_value)
        return all_bets

    def to_dataframe(self, bets: list[ValueBet]) -> pd.DataFrame:
        """Converte lista de ValueBets em DataFrame."""
        if not bets:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "match": f"{b.home_team} vs {b.away_team}",
                "market": b.market,
                "model_prob": b.model_prob,
                "implied_prob": b.implied_prob,
                "odd": b.odd,
                "edge": b.edge,
                "ev": b.expected_value,
                "stars": b.confidence_stars,
            }
            for b in bets
        ])
