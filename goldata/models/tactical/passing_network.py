"""Análise de redes de passe usando NetworkX."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx

from goldata.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PassingNetworkStats:
    """Métricas extraídas da rede de passe de um time."""
    team_id: str
    n_players: int
    n_edges: int
    avg_passes: float
    density: float
    most_central_player: str
    betweenness_centrality: dict[str, float] = field(default_factory=dict)
    degree_centrality: dict[str, float] = field(default_factory=dict)
    edge_weights: dict[str, float] = field(default_factory=dict)


class PassingNetworkAnalyzer:
    """
    Analisa a rede de passe de um time usando teoria de grafos.

    Métricas:
    - Centralidade de intermediação (quem conecta o jogo)
    - Centralidade de grau (quem toca mais na bola)
    - Densidade da rede (estilo de jogo coletivo vs individual)
    - Combinações mais frequentes (duplas / trios)
    """

    def build_network(
        self,
        events_df: pd.DataFrame,
        team_id: str,
        min_passes: int = 3,
    ) -> nx.DiGraph:
        """
        Constrói grafo dirigido de passes de um time.

        Args:
            events_df: DataFrame com event_type, team_id, player_id, outcome
            team_id: ID do time a analisar
            min_passes: mínimo de passes para incluir uma aresta

        Returns:
            DiGraph com jogadores como nós e passes como arestas ponderadas
        """
        passes = events_df[
            (events_df["event_type"] == "pass")
            & (events_df["team_id"] == team_id)
            & (events_df.get("outcome", pd.Series(["success"] * len(events_df))) == "success")
        ].copy()

        G = nx.DiGraph()

        if len(passes) == 0:
            return G

        # Conectar passes consecutivos do mesmo time
        players = passes["player_id"].tolist()
        for i in range(len(players) - 1):
            src = str(players[i])
            dst = str(players[i + 1])
            if G.has_edge(src, dst):
                G[src][dst]["weight"] += 1
            else:
                G.add_edge(src, dst, weight=1)

        # Remover arestas com poucos passes
        edges_to_remove = [
            (u, v) for u, v, d in G.edges(data=True) if d["weight"] < min_passes
        ]
        G.remove_edges_from(edges_to_remove)

        return G

    def analyze(
        self,
        events_df: pd.DataFrame,
        team_id: str,
        min_passes: int = 3,
    ) -> PassingNetworkStats:
        """
        Analisa a rede de passe e retorna métricas do time.
        """
        G = self.build_network(events_df, team_id, min_passes)

        if len(G.nodes) == 0:
            return PassingNetworkStats(
                team_id=team_id, n_players=0, n_edges=0,
                avg_passes=0.0, density=0.0, most_central_player="N/A",
            )

        betweenness = nx.betweenness_centrality(G, weight="weight")
        degree_cent = nx.degree_centrality(G)

        # Jogador mais central (betweenness)
        most_central = max(betweenness, key=betweenness.get) if betweenness else "N/A"

        # Peso médio das arestas
        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        avg_passes = float(np.mean(weights)) if weights else 0.0

        # Edge weights para as combinações principais
        edge_weights = {
            f"{u}->{v}": d["weight"]
            for u, v, d in sorted(G.edges(data=True), key=lambda x: -x[2]["weight"])[:10]
        }

        return PassingNetworkStats(
            team_id=team_id,
            n_players=len(G.nodes),
            n_edges=len(G.edges),
            avg_passes=round(avg_passes, 2),
            density=round(nx.density(G), 4),
            most_central_player=most_central,
            betweenness_centrality={k: round(v, 4) for k, v in betweenness.items()},
            degree_centrality={k: round(v, 4) for k, v in degree_cent.items()},
            edge_weights=edge_weights,
        )

    def get_top_combinations(
        self,
        G: nx.DiGraph,
        n: int = 5,
    ) -> list[dict[str, Any]]:
        """Retorna as N duplas de jogadores com mais passes entre si."""
        combos = [
            {"from": u, "to": v, "passes": d["weight"]}
            for u, v, d in G.edges(data=True)
        ]
        combos.sort(key=lambda x: -x["passes"])
        return combos[:n]
