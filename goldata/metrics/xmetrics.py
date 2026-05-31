"""Métricas avançadas: xT (Expected Threat), calibração, avaliação de modelos."""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    precision_recall_curve, average_precision_score,
)

from goldata.logging_config import get_logger

logger = get_logger(__name__)

# Grid xT: 12x8 zonas do campo
XT_GRID_X = 12
XT_GRID_Y = 8

# xT grid pré-calculado (baseado em Karun Singh 2019)
# Valores representam a probabilidade de gol se a posse de bola for nessa zona
_XT_BASE_GRID = np.array([
    [0.006, 0.006, 0.006, 0.006, 0.007, 0.008, 0.010, 0.012, 0.020, 0.040, 0.080, 0.150],
    [0.006, 0.006, 0.006, 0.007, 0.008, 0.010, 0.012, 0.018, 0.030, 0.060, 0.100, 0.200],
    [0.006, 0.007, 0.007, 0.008, 0.009, 0.011, 0.014, 0.021, 0.036, 0.070, 0.120, 0.250],
    [0.006, 0.007, 0.007, 0.008, 0.010, 0.012, 0.015, 0.024, 0.042, 0.080, 0.140, 0.300],
    [0.006, 0.007, 0.007, 0.008, 0.010, 0.012, 0.015, 0.024, 0.042, 0.080, 0.140, 0.300],
    [0.006, 0.007, 0.007, 0.008, 0.009, 0.011, 0.014, 0.021, 0.036, 0.070, 0.120, 0.250],
    [0.006, 0.006, 0.006, 0.007, 0.008, 0.010, 0.012, 0.018, 0.030, 0.060, 0.100, 0.200],
    [0.006, 0.006, 0.006, 0.006, 0.007, 0.008, 0.010, 0.012, 0.020, 0.040, 0.080, 0.150],
])  # shape: (8, 12): rows = y zones, cols = x zones

FIELD_LENGTH = 120.0
FIELD_WIDTH = 80.0


def _pos_to_xt_zone(x: float, y: float) -> tuple[int, int]:
    col = min(int(x / FIELD_LENGTH * XT_GRID_X), XT_GRID_X - 1)
    row = min(int(y / FIELD_WIDTH * XT_GRID_Y), XT_GRID_Y - 1)
    return row, col


class ExpectedThreat:
    """
    xT: Expected Threat (Karun Singh, 2019).

    Mede o valor de cada ação (passe, dribble, corrida) em termos
    de aumento na probabilidade de gol.

    xT(ação) = xT(destino) - xT(origem)
    """

    def __init__(self) -> None:
        self._grid = _XT_BASE_GRID.copy()

    def get_xt_value(self, x: float, y: float) -> float:
        """Retorna valor xT de uma posição."""
        row, col = _pos_to_xt_zone(x, y)
        return float(self._grid[row, col])

    def calculate_xt_action(
        self,
        start_x: float, start_y: float,
        end_x: float, end_y: float,
    ) -> float:
        """
        Calcula o xT gerado por uma ação (passe, dribble, etc).

        Retorna valor positivo = ação ameaçadora, negativo = recuo seguro.
        """
        xt_start = self.get_xt_value(start_x, start_y)
        xt_end = self.get_xt_value(end_x, end_y)
        return round(xt_end - xt_start, 6)

    def calculate_xt_per_player(
        self,
        events_df: pd.DataFrame,
        player_col: str = "player_id",
    ) -> pd.DataFrame:
        """
        Calcula xT total e por 90min gerado por cada jogador.

        Args:
            events_df: DataFrame com player_id, x, y, end_x, end_y

        Returns:
            DataFrame com player_id, xt_total, xt_per_action
        """
        required = ["x", "y", "end_x", "end_y"]
        for col in required:
            if col not in events_df.columns:
                events_df = events_df.copy()
                events_df[col] = 0.0

        events_df = events_df.copy()
        events_df["xt"] = events_df.apply(
            lambda r: self.calculate_xt_action(r["x"], r["y"], r["end_x"], r["end_y"]),
            axis=1,
        )

        grouped = events_df.groupby(player_col)["xt"].agg(
            xt_total="sum",
            xt_per_action="mean",
            n_actions="count",
        ).reset_index()

        return grouped.sort_values("xt_total", ascending=False).reset_index(drop=True)

    def get_grid(self) -> np.ndarray:
        """Retorna o grid de xT (8x12)."""
        return self._grid.copy()


class EvaluationMetrics:
    """Métricas de avaliação de modelos de classificação esportiva."""

    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        Calcula todas as métricas relevantes para modelos de xG/previsão.

        Args:
            y_true: array de 0s e 1s
            y_pred: array de probabilidades [0, 1]

        Returns:
            dict com auc, log_loss, brier_score, avg_precision
        """
        y_true = np.array(y_true)
        y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)

        return {
            "auc": round(float(roc_auc_score(y_true, y_pred)), 4),
            "log_loss": round(float(log_loss(y_true, y_pred)), 4),
            "brier_score": round(float(brier_score_loss(y_true, y_pred)), 4),
            "avg_precision": round(float(average_precision_score(y_true, y_pred)), 4),
            "n_samples": len(y_true),
            "positive_rate": round(float(y_true.mean()), 4),
            "mean_predicted_prob": round(float(y_pred.mean()), 4),
        }

    @staticmethod
    def calibration_analysis(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Analisa calibração do modelo: compara prob prevista com freq real.

        Returns:
            DataFrame com prob_mean, fraction_positive por bin
        """
        y_true = np.array(y_true)
        y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)
        frac_pos, prob_mean = calibration_curve(y_true, y_pred, n_bins=n_bins)
        return pd.DataFrame({
            "predicted_prob": prob_mean.round(4),
            "actual_fraction": frac_pos.round(4),
            "calibration_error": (prob_mean - frac_pos).round(4),
        })

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Expected Calibration Error (ECE): métrica compacta de calibração.
        0 = perfeitamente calibrado.
        """
        df = EvaluationMetrics.calibration_analysis(y_true, y_pred, n_bins)
        n = len(y_true)
        bin_sizes = np.histogram(y_pred, bins=n_bins, range=(0, 1))[0]
        weights = bin_sizes / n if n > 0 else np.zeros(n_bins)
        weights = weights[:len(df)]
        ece = float(np.sum(weights * np.abs(df["calibration_error"].values)))
        return round(ece, 4)
