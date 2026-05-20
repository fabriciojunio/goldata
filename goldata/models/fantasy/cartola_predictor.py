"""Otimizador de escalação para Cartola FC usando programação linear (PuLP)."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from goldata.logging_config import get_logger

logger = get_logger(__name__)

# Formação padrão Cartola FC 2024
CARTOLA_POSITIONS = {
    "GOL": 1,  # goleiro
    "LAT": 2,  # laterais
    "ZAG": 2,  # zagueiros
    "MEI": 3,  # meias
    "ATA": 3,  # atacantes
    "TEC": 1,  # técnico
}

CARTOLA_BUDGET_DEFAULT = 140.0  # cartoletas padrão


@dataclass
class CartolaPrediction:
    player_id: str
    display_name: str
    position: str
    team: str
    predicted_points: float
    price: float
    cost_efficiency: float  # pontos por cartoleta
    confidence: float


@dataclass
class CartolaDraftResult:
    players: list[CartolaPrediction]
    total_price: float
    predicted_total_points: float
    formation: str
    budget_remaining: float


class CartolaPredictor:
    """
    Prediz pontuação de jogadores no Cartola FC e otimiza escalação.

    Abordagem:
    1. Prever pontuação esperada por jogador (baseado em stats + adversário)
    2. Otimizar escalação dado budget e restrições de posição
    3. Calcular eficiência de custo (pontos/cartoleta)

    Se PuLP não estiver disponível, usa greedy por eficiência.
    """

    def __init__(self) -> None:
        self._predictions: list[CartolaPrediction] = []

    def predict_player_points(
        self,
        player_data: dict,
        opponent_defense_rating: float = 1.0,
    ) -> float:
        """
        Estima pontuação de um jogador no Cartola.

        Baseado em métricas recentes e dificuldade do adversário.

        Args:
            player_data: dict com xg_per_90, xa_per_90, minutes_avg, etc.
            opponent_defense_rating: 0.5=forte adversário, 1.5=adversário fraco

        Returns:
            Pontuação esperada (float)
        """
        position = str(player_data.get("position", "MEI")).upper()
        xg = float(player_data.get("xg_per_90", 0))
        xa = float(player_data.get("xa_per_90", 0))
        tackles = float(player_data.get("tackles_per_90", 0))
        interceptions = float(player_data.get("interceptions_per_90", 0))
        goals = float(player_data.get("goals_per_90", 0))
        assists = float(player_data.get("assists_per_90", 0))
        minutes = float(player_data.get("minutes_last_match", 90))
        form = float(player_data.get("form_last_3", 5.0))  # média pontos last 3

        # Pontos base por posição
        if position == "ATA":
            base = goals * 8 + assists * 5 + xg * 3 + xa * 2
        elif position == "MEI":
            base = goals * 7 + assists * 5 + xa * 3 + xg * 2
        elif position in ("LAT",):
            base = goals * 7 + assists * 5 + tackles * 1.5 + interceptions * 1.2
        elif position in ("ZAG",):
            base = tackles * 1.8 + interceptions * 1.5 + goals * 7
        elif position == "GOL":
            base = 3.0 + (1 - min(xg * 2, 1.0)) * 5  # pontos por defesas
        elif position == "TEC":
            base = goals * 0.5 + assists * 0.3 + 3.0
        else:
            base = xg * 3 + xa * 2

        # Ajuste por forma recente e adversário
        predicted = base * (minutes / 90.0) * opponent_defense_rating
        predicted = predicted * 0.7 + form * 0.3  # blend com forma

        return round(max(0.0, predicted), 2)

    def build_predictions(
        self,
        players_df: pd.DataFrame,
        opponent_ratings: dict[str, float] | None = None,
    ) -> list[CartolaPrediction]:
        """
        Gera previsões para todos os jogadores disponíveis.

        Args:
            players_df: DataFrame com player_id, display_name, position, team, price
            opponent_ratings: dict {team: defense_rating}

        Returns:
            Lista de CartolaPrediction ordenada por predicted_points desc
        """
        if opponent_ratings is None:
            opponent_ratings = {}

        predictions = []
        for _, row in players_df.iterrows():
            team = str(row.get("team", ""))
            opp_rating = opponent_ratings.get(team, 1.0)
            pts = self.predict_player_points(row.to_dict(), opp_rating)
            price = float(row.get("price", row.get("cartoletas", 5.0)))
            predictions.append(CartolaPrediction(
                player_id=str(row.get("player_id", "")),
                display_name=str(row.get("display_name", "Unknown")),
                position=str(row.get("position", "MEI")).upper(),
                team=team,
                predicted_points=pts,
                price=price,
                cost_efficiency=round(pts / max(price, 0.1), 4),
                confidence=0.7,
            ))

        self._predictions = sorted(predictions, key=lambda x: -x.predicted_points)
        return self._predictions

    def optimize_draft(
        self,
        budget: float = CARTOLA_BUDGET_DEFAULT,
        formation: dict[str, int] | None = None,
        excluded_players: list[str] | None = None,
    ) -> CartolaDraftResult:
        """
        Otimiza a escalação para maximizar pontos dentro do budget.

        Tenta usar PuLP (programação linear inteira). Se indisponível,
        usa algoritmo greedy por eficiência de custo.

        Args:
            budget: cartoletas disponíveis
            formation: posições e quantidades (padrão: CARTOLA_POSITIONS)
            excluded_players: IDs de jogadores a excluir (contusões, etc.)
        """
        if not self._predictions:
            logger.warning("no_predictions_available_run_build_predictions_first")
            return CartolaDraftResult(
                players=[], total_price=0.0,
                predicted_total_points=0.0, formation="4-3-3",
                budget_remaining=budget,
            )

        slots = formation or CARTOLA_POSITIONS
        excluded = set(excluded_players or [])
        available = [p for p in self._predictions if p.player_id not in excluded]

        # Tentar PuLP
        try:
            return self._optimize_pulp(available, budget, slots)
        except ImportError:
            logger.info("pulp_not_available_using_greedy")
            return self._optimize_greedy(available, budget, slots)

    def _optimize_greedy(
        self,
        players: list[CartolaPrediction],
        budget: float,
        slots: dict[str, int],
    ) -> CartolaDraftResult:
        """Greedy por eficiência de custo."""
        selected: list[CartolaPrediction] = []
        remaining_budget = budget
        remaining_slots = dict(slots)

        # Ordenar por eficiência de custo
        sorted_players = sorted(players, key=lambda x: -x.cost_efficiency)

        for player in sorted_players:
            pos = player.position
            if pos not in remaining_slots or remaining_slots[pos] <= 0:
                continue
            if player.price > remaining_budget:
                continue
            selected.append(player)
            remaining_budget -= player.price
            remaining_slots[pos] -= 1

            if all(v == 0 for v in remaining_slots.values()):
                break

        total_pts = sum(p.predicted_points for p in selected)
        total_price = sum(p.price for p in selected)
        return CartolaDraftResult(
            players=selected,
            total_price=round(total_price, 2),
            predicted_total_points=round(total_pts, 2),
            formation="4-3-3",
            budget_remaining=round(budget - total_price, 2),
        )

    def _optimize_pulp(
        self,
        players: list[CartolaPrediction],
        budget: float,
        slots: dict[str, int],
    ) -> CartolaDraftResult:
        """Otimização com PuLP (programação linear inteira)."""
        import pulp

        n = len(players)
        prob = pulp.LpProblem("cartola_draft", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

        # Objetivo: maximizar pontos
        prob += pulp.lpSum(players[i].predicted_points * x[i] for i in range(n))

        # Restrição de budget
        prob += pulp.lpSum(players[i].price * x[i] for i in range(n)) <= budget

        # Restrições de posição
        for pos, count in slots.items():
            prob += pulp.lpSum(x[i] for i in range(n) if players[i].position == pos) == count

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Se infeasible (ex: budget muito baixo), usar greedy
        if prob.status != 1:
            return self._optimize_greedy(players, budget, slots)

        selected = [players[i] for i in range(n)
                    if pulp.value(x[i]) is not None and pulp.value(x[i]) > 0.5]
        total_pts = sum(p.predicted_points for p in selected)
        total_price = sum(p.price for p in selected)
        return CartolaDraftResult(
            players=selected,
            total_price=round(total_price, 2),
            predicted_total_points=round(total_pts, 2),
            formation="4-3-3",
            budget_remaining=round(budget - total_price, 2),
        )

    def get_best_by_position(self, position: str, n: int = 5) -> list[CartolaPrediction]:
        """Retorna os N melhores por posição."""
        pos = position.upper()
        filtered = [p for p in self._predictions if p.position == pos]
        return sorted(filtered, key=lambda x: -x.predicted_points)[:n]
