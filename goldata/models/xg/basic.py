"""Modelo xG básico: Logistic Regression + Random Forest ensemble."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from goldata.models.base import BaseMLModel, TrainResult
from goldata.logging_config import get_logger

logger = get_logger(__name__)

XG_BASIC_FEATURES = [
    "distance_to_goal",
    "angle_to_goal",
    "is_header",
    "is_foot_right",
    "is_foot_left",
    "is_penalty",
    "is_direct_freekick",
    "is_open_play",
]


class BasicXGModel(BaseMLModel):
    """
    Modelo xG básico: ensemble de Logistic Regression e Random Forest.

    Inspirado nos modelos simples de xG usados por times de Série A
    sem grandes recursos computacionais. Funciona bem com poucos dados.

    Features: distância, ângulo, cabeceio, pé, pênalti, falta, jogo aberto.
    """

    model_name = "BasicXGModel"
    model_version = "1.0.0"

    def __init__(self, random_state: int = 42) -> None:
        super().__init__()
        self.random_state = random_state

        self._lr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=1.0, max_iter=500, random_state=random_state, solver="lbfgs"
            )),
        ])

        self._rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )

        self._lr_weight = 0.5
        self._rf_weight = 0.5

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Garante que todas as features necessárias existem."""
        X = X.copy()
        for feat in XG_BASIC_FEATURES:
            if feat not in X.columns:
                X[feat] = 0.0
        return X[XG_BASIC_FEATURES].fillna(0.0)

    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainResult:
        """
        Treina o ensemble LR + RF.

        Args:
            X: DataFrame com features de chute
            y: Series binária (1=gol, 0=não-gol)

        Returns:
            TrainResult com métricas de treinamento
        """
        X_prep = self._prepare_features(X)
        self._feature_names = XG_BASIC_FEATURES

        self._lr_pipeline.fit(X_prep, y)
        self._rf.fit(X_prep, y)

        self.is_trained = True

        y_proba = self.predict(X_prep)
        result = self._compute_train_result(
            self.model_name, y, y_proba, XG_BASIC_FEATURES
        )
        logger.info(
            "basic_xg_trained",
            auc=result.train_auc,
            n_samples=result.n_samples,
        )
        return result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna probabilidades de gol (ensemble LR + RF)."""
        self._check_trained()
        X_prep = self._prepare_features(X)
        lr_proba = self._lr_pipeline.predict_proba(X_prep)[:, 1]
        rf_proba = self._rf.predict_proba(X_prep)[:, 1]
        return (self._lr_weight * lr_proba + self._rf_weight * rf_proba)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna array [[prob_0, prob_1], ...] shape (n, 2)."""
        proba_1 = self.predict(X)
        proba_0 = 1 - proba_1
        return np.column_stack([proba_0, proba_1])

    def get_feature_importance(self) -> pd.DataFrame:
        """Importância das features pelo Random Forest."""
        self._check_trained()
        return pd.DataFrame({
            "feature": XG_BASIC_FEATURES,
            "importance": self._rf.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
