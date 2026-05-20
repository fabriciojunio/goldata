"""Testes do modelo xG avançado."""

import pytest
import numpy as np
import pandas as pd
from goldata.models.xg.advanced import AdvancedXGModel, XG_ADVANCED_FEATURES
from goldata.exceptions import ModelNotTrainedError


@pytest.fixture
def trained_advanced(sample_shots_df):
    model = AdvancedXGModel(random_state=42)
    X = sample_shots_df[[f for f in XG_ADVANCED_FEATURES if f in sample_shots_df.columns]]
    y = sample_shots_df["is_goal"]
    result = model.train(X, y)
    return model, sample_shots_df, result


def test_advanced_model_trains(sample_shots_df):
    model = AdvancedXGModel(random_state=42)
    X = sample_shots_df
    y = sample_shots_df["is_goal"]
    result = model.train(X, y)
    assert model.is_trained is True
    assert result.train_auc > 0


def test_advanced_predict_before_train_raises():
    model = AdvancedXGModel()
    df = pd.DataFrame({"x": [100], "y": [40]})
    with pytest.raises(ModelNotTrainedError):
        model.predict(df)


def test_advanced_predict_returns_probabilities(trained_advanced):
    model, df, _ = trained_advanced
    preds = model.predict(df)
    assert (preds >= 0).all()
    assert (preds <= 1).all()


def test_advanced_predict_proba_shape(trained_advanced):
    model, df, _ = trained_advanced
    proba = model.predict_proba(df)
    assert proba.shape[1] == 2


def test_advanced_proba_sums_to_one(trained_advanced):
    model, df, _ = trained_advanced
    proba = model.predict_proba(df)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_advanced_auc_reasonable(trained_advanced):
    model, df, result = trained_advanced
    assert result.train_auc >= 0.50


def test_advanced_feature_importance_exists(trained_advanced):
    model, _, _ = trained_advanced
    fi = model.get_feature_importance()
    assert len(fi) > 0


def test_advanced_shap_or_permutation_importance(trained_advanced):
    """get_shap_values deve retornar array sem error."""
    model, df, _ = trained_advanced
    X = df[[f for f in XG_ADVANCED_FEATURES if f in df.columns]].head(10)
    shap_vals = model.get_shap_values(X)
    assert shap_vals is not None


def test_advanced_model_handles_missing_features(trained_advanced):
    model, _, _ = trained_advanced
    X_minimal = pd.DataFrame({"x": [100, 110], "y": [40, 35]})
    preds = model.predict(X_minimal)
    assert len(preds) == 2


def test_advanced_penalty_vs_long_shot(trained_advanced):
    model, _, _ = trained_advanced
    penalty = pd.DataFrame([{"distance_to_goal": 11.0, "angle_to_goal": 0.6,
                              "is_header": 0, "is_foot_right": 1, "is_foot_left": 0,
                              "is_penalty": 1, "is_direct_freekick": 0, "is_open_play": 1,
                              "shot_sequence_length": 1, "x": 108.0, "y": 40.0}])
    long_shot = pd.DataFrame([{"distance_to_goal": 32.0, "angle_to_goal": 0.1,
                                "is_header": 0, "is_foot_right": 1, "is_foot_left": 0,
                                "is_penalty": 0, "is_direct_freekick": 0, "is_open_play": 1,
                                "shot_sequence_length": 8, "x": 88.0, "y": 20.0}])
    assert model.predict(penalty)[0] > model.predict(long_shot)[0]


def test_train_result_has_extra_metrics(trained_advanced):
    _, _, result = trained_advanced
    assert "lgbm_available" in result.extra_metrics
