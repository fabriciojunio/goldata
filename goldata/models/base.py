"""Classe base abstrata para todos os modelos de ML do GolData."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from goldata.exceptions import ModelNotTrainedError
from goldata.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TrainResult:
    """Resultado de treinamento de um modelo."""

    model_name: str
    train_auc: float
    train_log_loss: float
    n_samples: int
    n_features: int
    feature_names: list[str] = field(default_factory=list)
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.model_name} | "
            f"AUC: {self.train_auc:.4f} | "
            f"LogLoss: {self.train_log_loss:.4f} | "
            f"n={self.n_samples}"
        )


class BaseMLModel(ABC):
    """
    Classe base para todos os modelos de ML do GolData.

    Todos os modelos devem:
    - Implementar train(), predict(), predict_proba()
    - Suportar save()/load() via joblib
    - Expor get_feature_importance()
    - Ter is_trained controlando se pode usar predict
    """

    model_name: str = "BaseModel"
    model_version: str = "1.0.0"

    def __init__(self) -> None:
        self.is_trained: bool = False
        self._model: Any = None
        self._feature_names: list[str] = []

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> "TrainResult":
        """Treina o modelo. Deve setar self.is_trained = True ao final."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna array de probabilidades [0, 1]."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna array [[prob_0, prob_1], ...] shape (n, 2)."""
        ...

    def _check_trained(self) -> None:
        """Lança exceção se modelo não foi treinado."""
        if not self.is_trained:
            raise ModelNotTrainedError(self.model_name)

    def get_feature_importance(self) -> pd.DataFrame:
        """Retorna DataFrame com importância das features."""
        self._check_trained()
        if hasattr(self._model, "feature_importances_"):
            return pd.DataFrame({
                "feature": self._feature_names,
                "importance": self._model.feature_importances_,
            }).sort_values("importance", ascending=False)
        if hasattr(self._model, "coef_"):
            return pd.DataFrame({
                "feature": self._feature_names,
                "importance": abs(self._model.coef_[0]),
            }).sort_values("importance", ascending=False)
        return pd.DataFrame({"feature": self._feature_names, "importance": 0.0})

    def save(self, path: str) -> None:
        """Salva modelo em disco usando joblib."""
        self._check_trained()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("model_saved", model=self.model_name, path=path)

    @classmethod
    def load(cls, path: str) -> "BaseMLModel":
        """Carrega modelo de disco."""
        model = joblib.load(path)
        logger.info("model_loaded", model=model.model_name, path=path)
        return model  # type: ignore[return-value]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Avalia o modelo com métricas padrão."""
        self._check_trained()
        proba = self.predict(X)
        return {
            "auc": float(roc_auc_score(y, proba)),
            "log_loss": float(log_loss(y, proba)),
            "brier_score": float(brier_score_loss(y, proba)),
        }

    @staticmethod
    def _compute_train_result(
        model_name: str,
        y_true: pd.Series,
        y_proba: np.ndarray,
        feature_names: list[str],
        extra: dict[str, Any] | None = None,
    ) -> TrainResult:
        auc = float(roc_auc_score(y_true, y_proba))
        ll = float(log_loss(y_true, y_proba))
        return TrainResult(
            model_name=model_name,
            train_auc=auc,
            train_log_loss=ll,
            n_samples=len(y_true),
            n_features=len(feature_names),
            feature_names=feature_names,
            extra_metrics=extra or {},
        )
