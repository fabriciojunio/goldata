"""Cliente de dados do Brasileirão Série A e B."""

import os
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from goldata.logging_config import get_logger
from goldata.cache import CacheManager

logger = get_logger(__name__)

BRASILEIRAO_QUALITY_FACTOR = 0.85  # Relativo à Ligue 1
HOME_ADVANTAGE_DEFAULT = 1.35

# Dados sample embutidos para quando arquivos não existem
_SAMPLE_TEAMS = [
    "Flamengo", "Palmeiras", "Atletico-MG", "Botafogo", "Fluminense",
    "Corinthians", "Sao Paulo", "Internacional", "Vasco", "Cruzeiro",
    "Bahia", "Fortaleza", "Atletico-GO", "Cuiaba", "Coritiba",
    "Goias", "America-MG", "Santos", "Gremio", "Bragantino"
]


class BrasileiraoDataClient:
    """
    Cliente de dados do Brasileirão Série A e B.

    Fontes (em ordem de prioridade):
    1. Arquivos locais em data/sample/brasileirao/
    2. Dataset aberto do GitHub (adaoduque/Brasileirao_Dataset)
    3. Dados sintéticos sample para testes
    """

    def __init__(self, data_dir: str = "./data") -> None:
        self.data_dir = Path(data_dir)
        self.sample_dir = self.data_dir / "sample" / "brasileirao"
        self.cache = CacheManager()
        self._ensure_sample_data()

    def _ensure_sample_data(self) -> None:
        """Cria arquivos sample se não existirem."""
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        standings_path = self.sample_dir / "serie_a_standings_2024.csv"
        if not standings_path.exists():
            self._create_sample_standings(standings_path)

        player_stats_path = self.sample_dir / "serie_a_player_stats_2024.csv"
        if not player_stats_path.exists():
            self._create_sample_player_stats(player_stats_path)

        matches_path = self.sample_dir / "serie_a_matches_2024.csv"
        if not matches_path.exists():
            self._create_sample_matches(matches_path)

    def _create_sample_standings(self, path: Path) -> None:
        np.random.seed(42)
        n = 20
        teams = _SAMPLE_TEAMS.copy()
        points = sorted(np.random.randint(20, 80, n), reverse=True)
        wins = [int(p * 0.32) for p in points]
        draws = [int(p * 0.15) for p in points]
        losses = [38 - w - d for w, d in zip(wins, draws)]
        data = {
            "position": range(1, n + 1),
            "team": teams,
            "points": points,
            "matches": [38] * n,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_for": np.random.randint(30, 75, n),
            "goals_against": np.random.randint(25, 65, n),
            "xg": np.random.uniform(28.0, 70.0, n).round(1),
            "xga": np.random.uniform(25.0, 65.0, n).round(1),
        }
        pd.DataFrame(data).to_csv(path, index=False)
        logger.info("sample_standings_created", path=str(path))

    def _create_sample_player_stats(self, path: Path) -> None:
        np.random.seed(42)
        n = 60
        positions = ["FW", "MF", "DF", "GK"]
        data = {
            "player_id": [f"br_player_{i}" for i in range(n)],
            "display_name": [f"Jogador Brasileiro {i}" for i in range(n)],
            "team": np.random.choice(_SAMPLE_TEAMS, n),
            "position": np.random.choice(positions, n, p=[0.25, 0.35, 0.30, 0.10]),
            "minutes_played": np.random.randint(500, 3200, n),
            "goals": np.random.randint(0, 20, n),
            "assists": np.random.randint(0, 12, n),
            "xg": np.random.uniform(0, 18, n).round(2),
            "xa": np.random.uniform(0, 10, n).round(2),
            "shots": np.random.randint(5, 110, n),
            "passes": np.random.randint(200, 2500, n),
            "passes_completed": np.random.randint(150, 2200, n),
            "key_passes": np.random.randint(3, 70, n),
            "progressive_passes": np.random.randint(8, 180, n),
            "tackles": np.random.randint(5, 90, n),
            "interceptions": np.random.randint(3, 55, n),
            "pressures": np.random.randint(30, 450, n),
            "dribbles": np.random.randint(3, 90, n),
            "dribbles_completed": np.random.randint(2, 70, n),
        }
        pd.DataFrame(data).to_csv(path, index=False)
        logger.info("sample_player_stats_created", path=str(path))

    def _create_sample_matches(self, path: Path) -> None:
        np.random.seed(42)
        matches = []
        teams = _SAMPLE_TEAMS
        for i in range(100):
            home = np.random.choice(teams)
            away = np.random.choice([t for t in teams if t != home])
            home_goals = np.random.poisson(1.6)
            away_goals = np.random.poisson(1.1)
            matches.append({
                "match_id": i + 1,
                "season": "2024",
                "round": np.random.randint(1, 39),
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_xg": round(np.random.uniform(0.5, 3.0), 2),
                "away_xg": round(np.random.uniform(0.3, 2.5), 2),
                "stadium": f"Estadio {home}",
                "attendance": np.random.randint(10000, 60000),
            })
        pd.DataFrame(matches).to_csv(path, index=False)
        logger.info("sample_matches_created", path=str(path))

    # ── API Pública ───────────────────────────────────────────────────────────

    def get_serie_a_standings(self, season: int = 2024) -> pd.DataFrame:
        """Retorna tabela de classificação da Série A."""
        cache_key = f"standings_{season}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        path = self.sample_dir / f"serie_a_standings_{season}.csv"
        if not path.exists():
            # Usar 2024 como fallback
            path = self.sample_dir / "serie_a_standings_2024.csv"

        df = pd.read_csv(path)
        self.cache.set(cache_key, df, ttl=3600)
        logger.info("standings_loaded", season=season, teams=len(df))
        return df

    def get_serie_a_player_stats(self, season: int = 2024) -> pd.DataFrame:
        """Retorna estatísticas de jogadores da Série A."""
        cache_key = f"player_stats_{season}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        path = self.sample_dir / f"serie_a_player_stats_{season}.csv"
        if not path.exists():
            path = self.sample_dir / "serie_a_player_stats_2024.csv"

        df = pd.read_csv(path)
        self.cache.set(cache_key, df, ttl=3600)
        logger.info("player_stats_loaded", season=season, players=len(df))
        return df

    def get_serie_a_matches(self, season: int = 2024) -> pd.DataFrame:
        """Retorna partidas da Série A com resultados e xG."""
        cache_key = f"matches_{season}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        path = self.sample_dir / f"serie_a_matches_{season}.csv"
        if not path.exists():
            path = self.sample_dir / "serie_a_matches_2024.csv"

        df = pd.read_csv(path)
        self.cache.set(cache_key, df, ttl=3600)
        return df

    def get_match_history(self, team_name: str, seasons: list[int] | None = None) -> pd.DataFrame:
        """Retorna histórico de partidas de um time."""
        if seasons is None:
            seasons = [2024]

        all_matches = []
        for season in seasons:
            df = self.get_serie_a_matches(season)
            team_matches = df[
                (df["home_team"] == team_name) | (df["away_team"] == team_name)
            ].copy()
            if len(team_matches) > 0:
                all_matches.append(team_matches)

        if not all_matches:
            return pd.DataFrame()
        return pd.concat(all_matches, ignore_index=True)

    def get_home_advantage_factor(self, team_name: str, seasons: list[int] | None = None) -> float:
        """Calcula fator de vantagem em casa de um time."""
        history = self.get_match_history(team_name, seasons)
        if len(history) == 0:
            return HOME_ADVANTAGE_DEFAULT

        home_games = history[history["home_team"] == team_name]
        away_games = history[history["away_team"] == team_name]

        if len(home_games) == 0 or len(away_games) == 0:
            return HOME_ADVANTAGE_DEFAULT

        home_ppg = (
            (home_games["home_goals"] > home_games["away_goals"]).sum() * 3
            + (home_games["home_goals"] == home_games["away_goals"]).sum()
        ) / len(home_games)

        away_ppg = (
            (away_games["away_goals"] > away_games["home_goals"]).sum() * 3
            + (away_games["away_goals"] == away_games["home_goals"]).sum()
        ) / len(away_games)

        if away_ppg > 0:
            return round(home_ppg / away_ppg, 3)
        return HOME_ADVANTAGE_DEFAULT

    @property
    def quality_factor(self) -> float:
        """Fator de qualidade do Brasileirão (comparado à Europa)."""
        return BRASILEIRAO_QUALITY_FACTOR
