"""Modelo Dixon-Coles: Poisson com correlação de placares baixos e decaimento temporal."""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from goldata.logging_config import get_logger

logger = get_logger(__name__)


def _rho_correction(home_goals: int, away_goals: int, lh: float, la: float, rho: float) -> float:
    """
    Fator de correção tau para placares baixos (Dixon & Coles, 1997).
    Corrige a independência suposta pelo Poisson bivariado.
    """
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lh * la * rho
    elif home_goals == 1 and away_goals == 0:
        return 1.0 + la * rho
    elif home_goals == 0 and away_goals == 1:
        return 1.0 + lh * rho
    elif home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


class DixonColes:
    """
    Modelo Dixon-Coles de previsão de resultados.

    Extensão do Poisson bivariado que:
    1. Corrige a correlação para placares baixos (fator rho/tau)
    2. Aplica decaimento temporal para dar mais peso a jogos recentes
    3. Estima parâmetros por MLE com scipy.optimize

    Referência: Dixon, M.J. e Coles, S.G. (1997).
    "Modelling Association Football Scores and Inefficiencies in the
    Football Betting Market." Applied Statistics, 46(2), 265–280.
    """

    def __init__(self, decay: float = 0.0065) -> None:
        """
        Args:
            decay: fator de decaimento temporal (default 0.0065 conforme paper)
        """
        self.decay = decay
        self._teams: list[str] = []
        self._params: dict[str, float] = {}
        self.is_trained: bool = False

    def _get_lambdas(self, home_team: str, away_team: str) -> tuple[float, float]:
        home_attack = self._params.get(f"attack_{home_team}", 1.0)
        home_defense = self._params.get(f"defense_{home_team}", 1.0)
        away_attack = self._params.get(f"attack_{away_team}", 1.0)
        away_defense = self._params.get(f"defense_{away_team}", 1.0)
        home_adv = self._params.get("home_advantage", 1.2)
        intercept = self._params.get("intercept", 1.0)

        lh = max(0.01, intercept * home_attack * away_defense * home_adv)
        la = max(0.01, intercept * away_attack * home_defense)
        return lh, la

    @staticmethod
    def _time_weight(days_ago: float, decay: float) -> float:
        """Peso temporal: exp(-decay * days_ago)."""
        return np.exp(-decay * days_ago)

    def fit(self, results_df: pd.DataFrame, decay: float | None = None) -> "DixonColes":
        """
        Estima parâmetros por MLE com decaimento temporal.

        Args:
            results_df: DataFrame com home_team, away_team, home_goals, away_goals
                        Opcional: coluna 'date' (ISO string) para decaimento temporal
            decay: sobrescreve o decay padrão se fornecido
        """
        if decay is not None:
            self.decay = decay

        teams = sorted(set(results_df["home_team"]) | set(results_df["away_team"]))
        self._teams = teams
        n_teams = len(teams)
        team_idx = {t: i for i, t in enumerate(teams)}

        # Calcular pesos temporais
        ref_date = datetime.now()
        weights = []
        for _, row in results_df.iterrows():
            try:
                match_date = datetime.fromisoformat(str(row.get("date", ref_date)))
                days_ago = max(0.0, (ref_date - match_date).days)
            except (ValueError, TypeError):
                days_ago = 0.0
            weights.append(self._time_weight(days_ago, self.decay))
        weights = np.array(weights)

        def neg_log_likelihood(params: np.ndarray) -> float:
            intercept = params[0]
            home_adv = params[1]
            rho = params[2]
            attack = params[3: 3 + n_teams]
            defense = params[3 + n_teams: 3 + 2 * n_teams]

            ll = 0.0
            for i, (_, row) in enumerate(results_df.iterrows()):
                hi = team_idx[row["home_team"]]
                ai = team_idx[row["away_team"]]
                hg = int(row["home_goals"])
                ag = int(row["away_goals"])

                lh = max(0.01, intercept * attack[hi] * defense[ai] * home_adv)
                la = max(0.01, intercept * attack[ai] * defense[hi])

                tau = _rho_correction(hg, ag, lh, la, rho)
                tau = max(0.001, tau)

                term = (
                    np.log(tau)
                    + poisson.logpmf(hg, lh)
                    + poisson.logpmf(ag, la)
                )
                ll += weights[i] * term

            return -ll if np.isfinite(ll) else 1e10

        x0 = np.concatenate([
            [1.0, 1.2, 0.1],
            np.ones(n_teams),
            np.ones(n_teams),
        ])
        bounds = (
            [(0.1, 5.0), (1.0, 2.0), (-0.5, 0.5)]
            + [(0.1, 5.0)] * n_teams
            + [(0.1, 5.0)] * n_teams
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                neg_log_likelihood,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500},
            )

        params = result.x
        self._params["intercept"] = float(params[0])
        self._params["home_advantage"] = float(params[1])
        self._params["rho"] = float(params[2])
        for i, team in enumerate(teams):
            self._params[f"attack_{team}"] = float(params[3 + i])
            self._params[f"defense_{team}"] = float(params[3 + n_teams + i])

        self.is_trained = True
        logger.info("dixon_coles_fitted", teams=n_teams, rho=round(float(params[2]), 4))
        return self

    def predict_match(self, home_team: str, away_team: str, max_goals: int = 7) -> dict:
        """Prediz probabilidades de placar e resultado."""
        lh, la = self._get_lambdas(home_team, away_team)
        rho = self._params.get("rho", 0.0)

        score_probs: dict[str, float] = {}
        home_win = 0.0
        draw = 0.0
        away_win = 0.0

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                tau = _rho_correction(h, a, lh, la, rho)
                tau = max(0.001, tau)
                prob = tau * poisson.pmf(h, lh) * poisson.pmf(a, la)
                score_probs[f"{h}x{a}"] = round(float(prob), 6)

                if h > a:
                    home_win += prob
                elif h == a:
                    draw += prob
                else:
                    away_win += prob

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": round(home_win, 4),
            "draw_prob": round(draw, 4),
            "away_win_prob": round(away_win, 4),
            "expected_home_goals": round(lh, 3),
            "expected_away_goals": round(la, 3),
            "rho": round(rho, 4),
        }

    def get_team_ratings(self) -> pd.DataFrame:
        records = [
            {
                "team": t,
                "attack": round(self._params.get(f"attack_{t}", 1.0), 4),
                "defense": round(self._params.get(f"defense_{t}", 1.0), 4),
            }
            for t in self._teams
        ]
        return pd.DataFrame(records).sort_values("attack", ascending=False).reset_index(drop=True)
