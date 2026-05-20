"""Motor de similaridade entre jogadores usando cosine similarity."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from goldata.exceptions import ModelNotTrainedError, DataNotFoundError
from goldata.logging_config import get_logger

logger = get_logger(__name__)

SIMILARITY_FEATURES = [
    "goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
    "shots_per_90", "key_passes_per_90", "progressive_passes_per_90",
    "tackles_per_90", "interceptions_per_90", "pressures_per_90",
    "dribbles_completed_per_90", "pass_completion_rate",
]


@dataclass
class SimilarPlayer:
    player_id: str
    display_name: str
    similarity_score: float
    position: str | None = None
    team: str | None = None


class PlayerSimilarityEngine:
    """
    Encontra jogadores similares usando cosine similarity.

    Use cases:
    - Encontrar substitutos baratos para jogadores caros
    - Scout de jogadores com perfil similar a referências conhecidas
    - Análise de mercado para transferências
    """

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._player_vectors: np.ndarray | None = None
        self._player_ids: list[str] = []
        self._player_names: dict[str, str] = {}
        self._player_positions: dict[str, str] = {}
        self._player_teams: dict[str, str] = {}
        self.is_trained: bool = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for feat in SIMILARITY_FEATURES:
            if feat not in result.columns:
                result[feat] = 0.0
        return result[SIMILARITY_FEATURES].fillna(0.0)

    def fit(self, stats_df: pd.DataFrame) -> "PlayerSimilarityEngine":
        """
        Indexa jogadores para busca de similaridade.

        Args:
            stats_df: DataFrame com player_id, display_name e métricas per_90
        """
        id_col = "player_id" if "player_id" in stats_df.columns else None
        name_col = "display_name" if "display_name" in stats_df.columns else None

        self._player_ids = (
            stats_df[id_col].astype(str).tolist() if id_col else [str(i) for i in range(len(stats_df))]
        )
        if name_col:
            self._player_names = dict(zip(self._player_ids, stats_df[name_col].astype(str)))
        if "position" in stats_df.columns:
            self._player_positions = dict(zip(self._player_ids, stats_df["position"].astype(str)))
        if "team" in stats_df.columns:
            self._player_teams = dict(zip(self._player_ids, stats_df["team"].astype(str)))

        X = self._prepare_features(stats_df)
        X_scaled = self._scaler.fit_transform(X)
        self._player_vectors = X_scaled
        self.is_trained = True

        logger.info("similarity_engine_fitted", n_players=len(self._player_ids))
        return self

    def find_similar(
        self,
        player_id: str,
        n: int = 10,
        same_position: bool = False,
        min_similarity: float = 0.5,
    ) -> list[SimilarPlayer]:
        """
        Retorna os N jogadores mais similares ao player_id dado.

        Args:
            player_id: ID do jogador de referência
            n: número máximo de resultados
            same_position: filtrar só mesma posição
            min_similarity: score mínimo de similaridade

        Returns:
            Lista de SimilarPlayer ordenada por similaridade decrescente
        """
        if not self.is_trained or self._player_vectors is None:
            raise ModelNotTrainedError("PlayerSimilarityEngine")

        if player_id not in self._player_ids:
            raise DataNotFoundError("Player", player_id)

        idx = self._player_ids.index(player_id)
        query_vec = self._player_vectors[idx].reshape(1, -1)
        scores = cosine_similarity(query_vec, self._player_vectors)[0]

        target_position = self._player_positions.get(player_id)
        results = []
        for i, (pid, score) in enumerate(zip(self._player_ids, scores)):
            if pid == player_id:
                continue
            if score < min_similarity:
                continue
            if same_position and target_position:
                if self._player_positions.get(pid) != target_position:
                    continue
            results.append(SimilarPlayer(
                player_id=pid,
                display_name=self._player_names.get(pid, pid),
                similarity_score=round(float(score), 4),
                position=self._player_positions.get(pid),
                team=self._player_teams.get(pid),
            ))

        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:n]

    def similarity_score(self, player_id_a: str, player_id_b: str) -> float:
        """Retorna o score de similaridade entre dois jogadores específicos."""
        if not self.is_trained or self._player_vectors is None:
            raise ModelNotTrainedError("PlayerSimilarityEngine")
        for pid in [player_id_a, player_id_b]:
            if pid not in self._player_ids:
                raise DataNotFoundError("Player", pid)
        idx_a = self._player_ids.index(player_id_a)
        idx_b = self._player_ids.index(player_id_b)
        vec_a = self._player_vectors[idx_a].reshape(1, -1)
        vec_b = self._player_vectors[idx_b].reshape(1, -1)
        return round(float(cosine_similarity(vec_a, vec_b)[0][0]), 4)

    def get_all_player_ids(self) -> list[str]:
        return self._player_ids.copy()
