"""Modelo xG posicional: probabilidade por zona do campo com suavização Bayesiana."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from goldata.models.base import BaseMLModel, TrainResult
from goldata.logging_config import get_logger

logger = get_logger(__name__)

# Grid do campo: 12 colunas x 8 linhas = 96 zonas
GRID_X = 12
GRID_Y = 8
FIELD_LENGTH = 120.0
FIELD_WIDTH = 80.0

# Prior Bayesiano: taxa média de gol histórica (~10%)
BAYESIAN_PRIOR_GOALS = 0.10
BAYESIAN_PRIOR_WEIGHT = 20  # peso do prior (equivale a 20 observações fictícias)


def xy_to_zone(x: float, y: float) -> tuple[int, int]:
    """Converte posição (x, y) em índices de zona (col, row)."""
    col = min(int(x / FIELD_LENGTH * GRID_X), GRID_X - 1)
    row = min(int(y / FIELD_WIDTH * GRID_Y), GRID_Y - 1)
    return col, row


class PositionalXGModel(BaseMLModel):
    """
    Modelo xG posicional baseado em zonas do campo.

    Divide o campo em 12x8 = 96 zonas e calcula a probabilidade de gol
    por zona usando suavização Bayesiana para zonas com poucos chutes.

    Útil como modelo baseline e para análise de regiões perigosas do campo.
    """

    model_name = "PositionalXGModel"
    model_version = "1.0.0"

    def __init__(self) -> None:
        super().__init__()
        # Matriz de xG por zona: shape (GRID_X, GRID_Y)
        self._xg_grid = np.full((GRID_X, GRID_Y), BAYESIAN_PRIOR_GOALS)
        self._shot_counts = np.zeros((GRID_X, GRID_Y))
        self._goal_counts = np.zeros((GRID_X, GRID_Y))

    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainResult:
        """
        Treina calculando taxa de gol por zona com suavização Bayesiana.

        Args:
            X: DataFrame com colunas 'x' e 'y'
            y: Series binária (1=gol, 0=não-gol)
        """
        self._feature_names = ["x", "y"]

        x_vals = X["x"].values if "x" in X.columns else X.iloc[:, 0].values
        y_vals = X["y"].values if "y" in X.columns else X.iloc[:, 1].values

        for xi, yi, goal in zip(x_vals, y_vals, y.values):
            col, row = xy_to_zone(float(xi), float(yi))
            self._shot_counts[col, row] += 1
            self._goal_counts[col, row] += int(goal)

        # Suavização Bayesiana
        for col in range(GRID_X):
            for row in range(GRID_Y):
                shots = self._shot_counts[col, row]
                goals = self._goal_counts[col, row]
                # Posterior: (prior_goals * prior_weight + observed_goals) / (prior_weight + shots)
                self._xg_grid[col, row] = (
                    BAYESIAN_PRIOR_GOALS * BAYESIAN_PRIOR_WEIGHT + goals
                ) / (BAYESIAN_PRIOR_WEIGHT + shots)

        self.is_trained = True

        y_proba = self.predict(X)
        result = self._compute_train_result(
            self.model_name, y, y_proba, self._feature_names
        )
        logger.info(
            "positional_xg_trained",
            auc=result.train_auc,
            n_samples=result.n_samples,
            total_zones_with_data=int((self._shot_counts > 0).sum()),
        )
        return result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna xG por zona para cada chute."""
        self._check_trained()
        x_vals = X["x"].values if "x" in X.columns else X.iloc[:, 0].values
        y_vals = X["y"].values if "y" in X.columns else X.iloc[:, 1].values

        probas = []
        for xi, yi in zip(x_vals, y_vals):
            col, row = xy_to_zone(float(xi), float(yi))
            probas.append(self._xg_grid[col, row])
        return np.array(probas)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba_1 = self.predict(X)
        return np.column_stack([1 - proba_1, proba_1])

    def get_xg_grid(self) -> np.ndarray:
        """Retorna a matriz de xG por zona (12x8)."""
        self._check_trained()
        return self._xg_grid.copy()

    def get_zone_heatmap(self) -> np.ndarray:
        """Alias para get_xg_grid — retorna heatmap de xG por zona."""
        return self.get_xg_grid()

    def get_xg_for_position(self, x: float, y: float) -> float:
        """Retorna xG para uma posição específica."""
        self._check_trained()
        col, row = xy_to_zone(x, y)
        return float(self._xg_grid[col, row])

    def get_feature_importance(self) -> pd.DataFrame:
        """Não aplicável para modelo posicional — retorna DataFrame vazio."""
        return pd.DataFrame({"feature": ["x", "y"], "importance": [0.5, 0.5]})
