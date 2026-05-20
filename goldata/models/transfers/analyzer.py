"""Análise de oportunidades de transferência: undervalued players e gap analysis."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from goldata.models.scouting.valuation import PlayerValuationModel
from goldata.models.scouting.similarity import PlayerSimilarityEngine
from goldata.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TransferOpportunity:
    player_id: str
    display_name: str
    position: str
    estimated_value: float     # M€ estimado pelo modelo
    market_value: float        # M€ de mercado (Transfermarkt-like)
    value_gap: float           # estimado - mercado (positivo = subvalorizado)
    value_gap_pct: float       # gap como % do mercado
    similar_players: list[str]
    recommendation: str        # "Comprar", "Vender", "Observar", "Manter"


class TransferAnalyzer:
    """
    Identifica oportunidades de mercado baseado em discrepâncias entre
    valor estimado pelo modelo e valor de mercado declarado.

    Use cases:
    - Encontrar jogadores subvalorizados (Moneyball approach)
    - Identificar quando vender antes do declínio
    - Comparar custo de reposição vs perfis similares disponíveis
    """

    def __init__(
        self,
        valuation_model: PlayerValuationModel | None = None,
        similarity_engine: PlayerSimilarityEngine | None = None,
    ) -> None:
        self._valuation = valuation_model
        self._similarity = similarity_engine

    def find_undervalued(
        self,
        players_df: pd.DataFrame,
        market_value_col: str = "market_value_m",
        min_gap_pct: float = 0.15,
        max_results: int = 20,
    ) -> list[TransferOpportunity]:
        """
        Identifica jogadores cujo valor de performance supera o valor de mercado.

        Args:
            players_df: DataFrame com métricas e market_value_m
            market_value_col: nome da coluna de valor de mercado
            min_gap_pct: gap mínimo (% do mercado) para considerar oportunidade
            max_results: máximo de oportunidades retornadas

        Returns:
            Lista de TransferOpportunity ordenada por gap decrescente
        """
        if market_value_col not in players_df.columns:
            players_df = players_df.copy()
            players_df[market_value_col] = 5.0  # fallback

        opportunities = []
        for _, row in players_df.iterrows():
            market_val = float(row.get(market_value_col, 5.0))

            if self._valuation and self._valuation.is_trained:
                estimated_val = float(self._valuation.predict(pd.DataFrame([row.to_dict()]))[0])
            else:
                # Fallback simples baseado em xG e assists
                xg = float(row.get("xg_per_90", 0)) if "xg_per_90" in row.index else float(row.get("xg", 0))
                xa = float(row.get("xa_per_90", 0)) if "xa_per_90" in row.index else float(row.get("xa", 0))
                age = float(row.get("age", 26))
                estimated_val = (xg * 20 + xa * 12 + max(0, (30 - age)) * 0.3)
                estimated_val = max(0.5, estimated_val)

            gap = estimated_val - market_val
            gap_pct = gap / market_val if market_val > 0 else 0.0

            if gap_pct >= min_gap_pct:
                pid = str(row.get("player_id", ""))
                similar = []
                if self._similarity and self._similarity.is_trained and pid:
                    try:
                        sims = self._similarity.find_similar(pid, n=3)
                        similar = [s.display_name for s in sims]
                    except Exception:
                        pass

                rec = "Comprar" if gap_pct >= 0.30 else "Observar"
                opportunities.append(TransferOpportunity(
                    player_id=pid,
                    display_name=str(row.get("display_name", "Unknown")),
                    position=str(row.get("position", "?")),
                    estimated_value=round(estimated_val, 2),
                    market_value=round(market_val, 2),
                    value_gap=round(gap, 2),
                    value_gap_pct=round(gap_pct, 4),
                    similar_players=similar,
                    recommendation=rec,
                ))

        opportunities.sort(key=lambda x: -x.value_gap_pct)
        return opportunities[:max_results]

    def find_overvalued(
        self,
        players_df: pd.DataFrame,
        market_value_col: str = "market_value_m",
        max_results: int = 10,
    ) -> list[TransferOpportunity]:
        """Identifica jogadores cujo valor de mercado supera o estimado (vender)."""
        if market_value_col not in players_df.columns:
            players_df = players_df.copy()
            players_df[market_value_col] = 5.0

        opportunities = []
        for _, row in players_df.iterrows():
            market_val = float(row.get(market_value_col, 5.0))

            if self._valuation and self._valuation.is_trained:
                estimated_val = float(self._valuation.predict(pd.DataFrame([row.to_dict()]))[0])
            else:
                xg = float(row.get("xg_per_90", 0)) if "xg_per_90" in row.index else float(row.get("xg", 0))
                xa = float(row.get("xa_per_90", 0)) if "xa_per_90" in row.index else float(row.get("xa", 0))
                age = float(row.get("age", 26))
                estimated_val = max(0.5, xg * 20 + xa * 12 + max(0, (30 - age)) * 0.3)

            gap = estimated_val - market_val
            gap_pct = gap / market_val if market_val > 0 else 0.0

            if gap_pct < -0.20:  # 20% abaixo do valor de mercado
                opportunities.append(TransferOpportunity(
                    player_id=str(row.get("player_id", "")),
                    display_name=str(row.get("display_name", "Unknown")),
                    position=str(row.get("position", "?")),
                    estimated_value=round(estimated_val, 2),
                    market_value=round(market_val, 2),
                    value_gap=round(gap, 2),
                    value_gap_pct=round(gap_pct, 4),
                    similar_players=[],
                    recommendation="Vender",
                ))

        opportunities.sort(key=lambda x: x.value_gap_pct)
        return opportunities[:max_results]

    def replacement_cost_analysis(
        self,
        player_id: str,
        players_df: pd.DataFrame,
        budget_m: float = 10.0,
        market_value_col: str = "market_value_m",
    ) -> pd.DataFrame:
        """
        Analisa opções de reposição dentro de um orçamento.

        Returns:
            DataFrame com jogadores similares acessíveis dentro do budget
        """
        if self._similarity is None or not self._similarity.is_trained:
            return pd.DataFrame()

        try:
            similar = self._similarity.find_similar(player_id, n=20, same_position=True)
        except Exception:
            return pd.DataFrame()

        similar_ids = [s.player_id for s in similar]
        similar_df = players_df[players_df.get("player_id", pd.Series()).isin(similar_ids)].copy() if "player_id" in players_df.columns else pd.DataFrame()

        if len(similar_df) == 0:
            return pd.DataFrame(
                [{"player_id": s.player_id, "display_name": s.display_name,
                  "similarity": s.similarity_score}
                 for s in similar]
            )

        affordable = similar_df[similar_df.get(market_value_col, 0) <= budget_m]
        return affordable
