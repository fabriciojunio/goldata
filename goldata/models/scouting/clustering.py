"""Clustering de jogadores por perfil de jogo — K-Means com k=8."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from goldata.exceptions import ModelNotTrainedError, InsufficientDataError
from goldata.logging_config import get_logger

logger = get_logger(__name__)

# Perfis pré-definidos (k=8 clusters representando arquétipos de jogadores)
ARCHETYPE_LABELS = {
    0: "Target Striker",       # centroavante fixo, muitos gols, pouca mobilidade
    1: "Creative Playmaker",   # camisa 10, muitos key passes e xa
    2: "Box-to-Box Midfielder",# cobre campo todo, tackles + key passes
    3: "Deep-Lying Playmaker", # pivô, muitos passes progressivos, pouco ataque
    4: "Wide Attacker",        # extremo, dribbles + aceleração + cruzamentos
    5: "Ball-Winning Midfielder",# volante, tackles + interceptions + pressures
    6: "Modern Fullback",      # lateral ofensivo, cruzamentos + dribbles
    7: "Defensive Anchor",     # zagueiro/volante defensivo, clearances + blocks
}

CLUSTERING_FEATURES = [
    "goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
    "shots_per_90", "key_passes_per_90", "progressive_passes_per_90",
    "tackles_per_90", "interceptions_per_90", "pressures_per_90",
    "dribbles_completed_per_90", "pass_completion_rate",
]


@dataclass
class ClusterProfile:
    cluster_id: int
    archetype: str
    n_players: int
    centroid: dict[str, float]
    top_features: list[str]


class PlayerClusterer:
    """
    Agrupa jogadores em 8 arquétipos usando K-Means.

    Permite identificar perfis de jogo similares,
    facilitando scouting e comparação entre jogadores.
    """

    def __init__(self, n_clusters: int = 8, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._kmeans: KMeans | None = None
        self._feature_names: list[str] = []
        self.is_trained: bool = False
        self._cluster_profiles: list[ClusterProfile] = []

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Garante que todas as features existem, preenche ausentes com 0."""
        result = df.copy()
        for feat in CLUSTERING_FEATURES:
            if feat not in result.columns:
                result[feat] = 0.0
        return result[CLUSTERING_FEATURES].fillna(0.0)

    def fit(self, stats_df: pd.DataFrame) -> "PlayerClusterer":
        """
        Treina o clusterer em dados de jogadores.

        Args:
            stats_df: DataFrame com métricas per_90 de jogadores

        Returns:
            self (para chaining)
        """
        if len(stats_df) < self.n_clusters:
            raise InsufficientDataError(
                minimum=self.n_clusters,
                received=len(stats_df),
                context="K-Means requer pelo menos n_clusters jogadores",
            )

        X = self._prepare_features(stats_df)
        self._feature_names = CLUSTERING_FEATURES

        X_scaled = self._scaler.fit_transform(X)
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
        )
        self._kmeans.fit(X_scaled)
        self.is_trained = True

        self._build_profiles(X, self._kmeans.labels_)
        logger.info(
            "player_clusterer_trained",
            n_clusters=self.n_clusters,
            n_players=len(stats_df),
        )
        return self

    def _build_profiles(self, X: pd.DataFrame, labels: np.ndarray) -> None:
        """Constrói perfis descritivos de cada cluster."""
        self._cluster_profiles = []
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            cluster_data = X[mask]
            if len(cluster_data) == 0:
                continue

            centroid = cluster_data.mean().round(3).to_dict()
            # Top 3 features acima da média geral
            overall_mean = X.mean()
            diff = (cluster_data.mean() - overall_mean).sort_values(ascending=False)
            top_features = diff.head(3).index.tolist()

            self._cluster_profiles.append(ClusterProfile(
                cluster_id=cluster_id,
                archetype=ARCHETYPE_LABELS.get(cluster_id, f"Profile {cluster_id}"),
                n_players=int(mask.sum()),
                centroid=centroid,
                top_features=top_features,
            ))

    def predict(self, stats_df: pd.DataFrame) -> np.ndarray:
        """Retorna cluster ID para cada jogador."""
        if not self.is_trained or self._kmeans is None:
            raise ModelNotTrainedError("PlayerClusterer")
        X = self._prepare_features(stats_df)
        X_scaled = self._scaler.transform(X)
        return self._kmeans.predict(X_scaled)

    def predict_with_distance(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Retorna cluster e distância ao centróide para cada jogador."""
        if not self.is_trained or self._kmeans is None:
            raise ModelNotTrainedError("PlayerClusterer")
        X = self._prepare_features(stats_df)
        X_scaled = self._scaler.transform(X)
        clusters = self._kmeans.predict(X_scaled)
        distances = self._kmeans.transform(X_scaled).min(axis=1)
        result = stats_df.copy()
        result["cluster_id"] = clusters
        result["cluster_archetype"] = [ARCHETYPE_LABELS.get(c, f"Profile {c}") for c in clusters]
        result["distance_to_centroid"] = distances.round(4)
        return result

    def get_cluster_profiles(self) -> list[ClusterProfile]:
        """Retorna perfis de cada cluster."""
        if not self.is_trained:
            raise ModelNotTrainedError("PlayerClusterer")
        return self._cluster_profiles

    def get_inertia(self) -> float:
        """Retorna inércia do K-Means (menor = melhor)."""
        if not self.is_trained or self._kmeans is None:
            raise ModelNotTrainedError("PlayerClusterer")
        return float(self._kmeans.inertia_)
