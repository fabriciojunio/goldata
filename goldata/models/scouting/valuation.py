"""Estimativa de valor de mercado de jogadores com GradientBoosting."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from goldata.models.base import BaseMLModel, TrainResult
from goldata.exceptions import ModelNotTrainedError
from goldata.logging_config import get_logger

logger = get_logger(__name__)

VALUATION_FEATURES = [
    "age", "goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
    "minutes_played", "shots_per_90", "key_passes_per_90",
    "progressive_passes_per_90", "tackles_per_90", "pressures_per_90",
    "dribbles_completed_per_90", "pass_completion_rate",
]


class PlayerValuationModel:
    """
    Estima valor de mercado de jogadores em milhões de euros.

    Baseado em métricas de performance e idade.
    Usa GradientBoostingRegressor com cross-validation.
    """

    model_name = "PlayerValuationModel"
    model_version = "1.0.0"

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=random_state,
        )
        self.is_trained: bool = False
        self._feature_names: list[str] = []

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if "age" not in result.columns:
            result["age"] = 25.0
        for feat in VALUATION_FEATURES:
            if feat not in result.columns:
                result[feat] = 0.0
        return result[VALUATION_FEATURES].fillna(0.0)

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Treina o modelo de valuation.

        Args:
            X: DataFrame com features dos jogadores
            y: Series com valores de mercado em M€

        Returns:
            dict com métricas de treino
        """
        X_prep = self._prepare_features(X)
        self._feature_names = VALUATION_FEATURES
        X_scaled = self._scaler.fit_transform(X_prep)
        self._model.fit(X_scaled, y)
        self.is_trained = True

        preds = self._model.predict(X_scaled)
        mae = float(np.mean(np.abs(preds - y.values)))
        rmse = float(np.sqrt(np.mean((preds - y.values) ** 2)))

        logger.info("valuation_model_trained", mae=mae, n_players=len(X))
        return {"mae_millions": round(mae, 3), "rmse_millions": round(rmse, 3), "n_samples": len(X)}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna estimativas de valor de mercado em M€."""
        if not self.is_trained:
            raise ModelNotTrainedError(self.model_name)
        X_prep = self._prepare_features(X)
        X_scaled = self._scaler.transform(X_prep)
        preds = self._model.predict(X_scaled)
        return np.maximum(preds, 0.0)  # valor de mercado não pode ser negativo

    def predict_single(self, player_data: dict) -> float:
        """Estima valor de um único jogador."""
        df = pd.DataFrame([player_data])
        return float(self.predict(df)[0])

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_trained:
            raise ModelNotTrainedError(self.model_name)
        return pd.DataFrame({
            "feature": self._feature_names,
            "importance": self._model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
