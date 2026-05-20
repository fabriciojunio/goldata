"""7 linhas finais para 100% de cobertura."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


# ══════════════════════════════════════════════════════════════════
# models/base.py linha 84: return DataFrame com feature_importances_
# ══════════════════════════════════════════════════════════════════

def test_base_get_feature_importance_with_feature_importances_():
    """Cobre linha 84: método base chamado com _model que tem feature_importances_."""
    from goldata.models.base import BaseMLModel, TrainResult
    from sklearn.ensemble import RandomForestClassifier

    class RFModel(BaseMLModel):
        model_name = "RFTest"
        def train(self, X, y):
            self._feature_names = list(X.columns)
            rf = RandomForestClassifier(n_estimators=10, random_state=0)
            rf.fit(X, y)
            self._model = rf  # RF tem feature_importances_ nativamente
            self.is_trained = True
            return TrainResult("RFTest", 0.7, 0.5, len(X), len(X.columns), list(X.columns))
        def predict(self, X): return np.zeros(len(X))
        def predict_proba(self, X): return np.zeros((len(X), 2))

    X = pd.DataFrame({"a": [0.1, 0.5, 0.9, 0.2, 0.8] * 10,
                      "b": [0.9, 0.1, 0.5, 0.7, 0.3] * 10})
    y = pd.Series([0, 1, 1, 0, 1] * 10)
    m = RFModel()
    m.train(X, y)
    # Não sobrescreve get_feature_importance → usa o da BaseMLModel (linha 84)
    fi = m.get_feature_importance()  # linha 84 do base.py
    assert isinstance(fi, pd.DataFrame)
    assert len(fi) == 2
    assert fi["importance"].sum() == pytest.approx(1.0, abs=0.01)


def test_base_get_feature_importance_coef_branch(sample_shots_df):
    """Cobre linha 89: modelo com coef_ (Logistic Regression no pipeline)."""
    from goldata.models.base import BaseMLModel, TrainResult
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    class LRModel(BaseMLModel):
        model_name = "LRTest"
        def train(self, X, y):
            self._feature_names = list(X.columns)
            pipe = Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(random_state=0, max_iter=200))])
            pipe.fit(X, y)
            self._model = pipe["lr"]  # LR tem coef_
            self.is_trained = True
            return TrainResult("LRTest", 0.6, 0.5, len(X), len(X.columns), list(X.columns))
        def predict(self, X): return np.zeros(len(X))
        def predict_proba(self, X): return np.zeros((len(X), 2))

    from goldata.models.xg.basic import XG_BASIC_FEATURES
    m = LRModel()
    X = sample_shots_df[XG_BASIC_FEATURES]
    y = sample_shots_df["is_goal"]
    m.train(X, y)

    # _model é um LR → tem coef_ mas não feature_importances_
    assert hasattr(m._model, "coef_")
    assert not hasattr(m._model, "feature_importances_")

    fi = m.get_feature_importance()  # linha 89
    assert len(fi) == len(XG_BASIC_FEATURES)


# ══════════════════════════════════════════════════════════════════
# models/scouting/clustering.py linha 155: get_cluster_profiles sem treino
# ══════════════════════════════════════════════════════════════════

def test_clustering_get_profiles_not_trained():
    """Cobre linha 155: get_cluster_profiles levanta quando não treinado."""
    from goldata.models.scouting.clustering import PlayerClusterer
    from goldata.exceptions import ModelNotTrainedError
    c = PlayerClusterer()
    c.is_trained = False  # explícito
    with pytest.raises(ModelNotTrainedError):
        c.get_cluster_profiles()


# ══════════════════════════════════════════════════════════════════
# models/scouting/similarity.py linha 139: similarity_score sem treino
# ══════════════════════════════════════════════════════════════════

def test_similarity_score_not_trained():
    """Cobre linha 139: similarity_score levanta quando não treinado."""
    from goldata.models.scouting.similarity import PlayerSimilarityEngine
    from goldata.exceptions import ModelNotTrainedError
    engine = PlayerSimilarityEngine()
    # is_trained=False e _player_vectors=None
    with pytest.raises(ModelNotTrainedError):
        engine.similarity_score("p1", "p2")


# ══════════════════════════════════════════════════════════════════
# models/transfers/analyzer.py linha 129: valuation.predict() chamado
# ══════════════════════════════════════════════════════════════════

def test_transfers_overvalued_with_valuation_model():
    """Cobre linha 129 via find_overvalued com valuation model treinado."""
    from goldata.models.transfers.analyzer import TransferAnalyzer
    from goldata.models.scouting.valuation import PlayerValuationModel

    feature_cols = ["goals_per_90","assists_per_90","xg_per_90","xa_per_90",
                    "shots_per_90","key_passes_per_90","progressive_passes_per_90",
                    "tackles_per_90","interceptions_per_90","pressures_per_90",
                    "dribbles_completed_per_90","pass_completion_rate"]
    np.random.seed(77)
    n = 30
    df = pd.DataFrame({
        "player_id": [f"ov{i}" for i in range(n)],
        "display_name": [f"OV{i}" for i in range(n)],
        "position": ["FW"] * n,
        "age": np.random.uniform(20, 34, n),
        "market_value_m": np.random.uniform(10.0, 50.0, n),  # alto
        **{f: np.random.uniform(0, 0.2, n) for f in feature_cols},  # baixo desempenho
    })
    val = PlayerValuationModel()
    val.train(df, df["market_value_m"])
    # Agora predict retornará valores baixos (linha 129 em find_overvalued)
    analyzer = TransferAnalyzer(valuation_model=val)
    opps = analyzer.find_overvalued(df)  # usa valuation para estimar
    assert isinstance(opps, list)


# ══════════════════════════════════════════════════════════════════
# models/xg/advanced.py linhas 157, 165-166: scorer + X_train None branch
# ══════════════════════════════════════════════════════════════════

def test_advanced_xg_shap_without_xtrain(sample_shots_df):
    """Cobre 165-166: branch quando _X_train é None no fallback de permutation."""
    import goldata.models.xg.advanced as adv
    from goldata.models.xg.advanced import XG_ADVANCED_FEATURES

    model = adv.AdvancedXGModel(random_state=42)
    X = sample_shots_df
    y = sample_shots_df["is_goal"]
    model.train(X, y)

    # Forçar _X_train = None para cobrir linhas 165-166
    model._X_train = None

    with patch.object(adv, "_SHAP_AVAILABLE", False):
        X_small = X.head(15)
        result = model.get_shap_values(X_small)
        assert result is not None
