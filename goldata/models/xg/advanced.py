"""Modelo xG avançado: XGBoost + LightGBM ensemble com SHAP."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from goldata.models.base import BaseMLModel, TrainResult
from goldata.logging_config import get_logger

logger = get_logger(__name__)

XG_ADVANCED_FEATURES = [
    "distance_to_goal",
    "angle_to_goal",
    "is_header",
    "is_foot_right",
    "is_foot_left",
    "is_penalty",
    "is_direct_freekick",
    "is_open_play",
    "shot_sequence_length",
    "x",
    "y",
]

# Tentar importar LightGBM; usar fallback XGBoost duplo se indisponível
try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False
    logger.warning("lgbm_unavailable_using_xgb_fallback")

# Tentar importar SHAP
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    from sklearn.inspection import permutation_importance
    _SHAP_AVAILABLE = False
    logger.warning("shap_unavailable_using_permutation_importance")


class AdvancedXGModel(BaseMLModel):
    """
    Modelo xG avançado: XGBoost + LightGBM ensemble.
    Usa SHAP para interpretabilidade (ou permutation importance como fallback).

    Features: tudo do básico + sequência de jogada, posição no campo.
    """

    model_name = "AdvancedXGModel"
    model_version = "1.0.0"

    def __init__(self, random_state: int = 42) -> None:
        super().__init__()
        self.random_state = random_state

        self._xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        )

        if _LGBM_AVAILABLE:
            self._lgbm = LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=31,
                random_state=random_state,
                verbose=-1,
            )
            self._xgb_weight = 0.55
            self._lgbm_weight = 0.45
        else:
            # Fallback: segundo XGBoost com hiperparâmetros diferentes
            self._lgbm = XGBClassifier(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.7,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=random_state + 1,
                verbosity=0,
            )
            self._xgb_weight = 0.5
            self._lgbm_weight = 0.5

        self._X_train: pd.DataFrame | None = None

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feat in XG_ADVANCED_FEATURES:
            if feat not in X.columns:
                X[feat] = 0.0
        return X[XG_ADVANCED_FEATURES].fillna(0.0)

    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainResult:
        X_prep = self._prepare_features(X)
        self._feature_names = XG_ADVANCED_FEATURES
        self._X_train = X_prep.copy()

        self._xgb.fit(X_prep, y)
        self._lgbm.fit(X_prep, y)

        self.is_trained = True

        y_proba = self.predict(X_prep)
        result = self._compute_train_result(
            self.model_name, y, y_proba, XG_ADVANCED_FEATURES,
            extra={"lgbm_available": _LGBM_AVAILABLE, "shap_available": _SHAP_AVAILABLE},
        )
        logger.info(
            "advanced_xg_trained",
            auc=result.train_auc,
            n_samples=result.n_samples,
            lgbm=_LGBM_AVAILABLE,
        )
        return result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_trained()
        X_prep = self._prepare_features(X)
        xgb_proba = self._xgb.predict_proba(X_prep)[:, 1]
        lgbm_proba = self._lgbm.predict_proba(X_prep)[:, 1]
        return self._xgb_weight * xgb_proba + self._lgbm_weight * lgbm_proba

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba_1 = self.predict(X)
        return np.column_stack([1 - proba_1, proba_1])

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calcula SHAP values para interpretabilidade.
        Se SHAP não disponível, usa permutation importance.
        """
        self._check_trained()
        X_prep = self._prepare_features(X)

        if _SHAP_AVAILABLE:
            explainer = shap.TreeExplainer(self._xgb)
            return explainer.shap_values(X_prep)
        else:
            from sklearn.inspection import permutation_importance as perm_imp

            # Usar X de treino se disponível
            if self._X_train is not None:
                n = min(100, len(self._X_train))
                X_eval = self._X_train.iloc[:n]
                y_eval = pd.Series(np.zeros(n))  # placeholder
            else:
                X_eval = X_prep
                y_eval = pd.Series(np.zeros(len(X_eval)))

            result = perm_imp(
                self._xgb, X_eval, y_eval,
                n_repeats=3, random_state=42
            )
            return result.importances

    def get_feature_importance(self) -> pd.DataFrame:
        self._check_trained()
        return pd.DataFrame({
            "feature": XG_ADVANCED_FEATURES,
            "importance": self._xgb.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


# Corrigir import faltando
from typing import Any
