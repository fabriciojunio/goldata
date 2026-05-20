"""Testes do modelo xG básico."""

import tempfile
import os
import pytest
import numpy as np
import pandas as pd
from goldata.models.xg.basic import BasicXGModel, XG_BASIC_FEATURES
from goldata.exceptions import ModelNotTrainedError
from goldata.data.features import FeatureEngineer


@pytest.fixture
def trained_model(sample_shots_df):
    model = BasicXGModel(random_state=42)
    X = sample_shots_df[XG_BASIC_FEATURES]
    y = sample_shots_df["is_goal"]
    model.train(X, y)
    return model, sample_shots_df


def test_model_trains_without_error(sample_shots_df):
    model = BasicXGModel(random_state=42)
    X = sample_shots_df[XG_BASIC_FEATURES]
    y = sample_shots_df["is_goal"]
    result = model.train(X, y)
    assert model.is_trained is True
    assert result is not None


def test_train_returns_train_result(sample_shots_df):
    model = BasicXGModel()
    X = sample_shots_df[XG_BASIC_FEATURES]
    y = sample_shots_df["is_goal"]
    result = model.train(X, y)
    assert hasattr(result, "train_auc")
    assert hasattr(result, "n_samples")


def test_predict_before_train_raises(sample_shots_df):
    model = BasicXGModel()
    with pytest.raises(ModelNotTrainedError):
        model.predict(sample_shots_df)


def test_predict_returns_array(trained_model):
    model, df = trained_model
    preds = model.predict(df[XG_BASIC_FEATURES])
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(df)


def test_predict_probabilities_between_0_and_1(trained_model):
    model, df = trained_model
    preds = model.predict(df[XG_BASIC_FEATURES])
    assert (preds >= 0).all()
    assert (preds <= 1).all()


def test_predict_proba_shape(trained_model):
    model, df = trained_model
    proba = model.predict_proba(df[XG_BASIC_FEATURES])
    assert proba.shape == (len(df), 2)


def test_predict_proba_sums_to_1(trained_model):
    model, df = trained_model
    proba = model.predict_proba(df[XG_BASIC_FEATURES])
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_penalty_xg_higher_than_long_shot():
    """Treina em dados controlados onde pênaltis têm taxa de gol maior."""
    np.random.seed(0)
    n = 500
    # Criar dataset onde pênaltis sempre entram (taxa 0.75) e chutes longe raramente (0.05)
    X = pd.DataFrame({
        "distance_to_goal": np.concatenate([np.full(250, 11.0), np.full(250, 32.0)]),
        "angle_to_goal": np.concatenate([np.full(250, 0.6), np.full(250, 0.15)]),
        "is_header": np.zeros(500),
        "is_foot_right": np.ones(500),
        "is_foot_left": np.zeros(500),
        "is_penalty": np.concatenate([np.ones(250), np.zeros(250)]),
        "is_direct_freekick": np.zeros(500),
        "is_open_play": np.ones(500),
    })
    y = pd.Series(np.concatenate([
        np.random.binomial(1, 0.75, 250),
        np.random.binomial(1, 0.05, 250)
    ]))
    model = BasicXGModel(random_state=0)
    model.train(X, y)
    penalty = X.iloc[:1]
    long_shot = X.iloc[250:251]
    assert model.predict(penalty)[0] > model.predict(long_shot)[0]


def test_auc_above_threshold(trained_model):
    model, df = trained_model
    metrics = model.evaluate(df[XG_BASIC_FEATURES], df["is_goal"])
    assert metrics["auc"] >= 0.55  # threshold conservador para dados sintéticos


def test_feature_importance_not_empty(trained_model):
    model, _ = trained_model
    fi = model.get_feature_importance()
    assert len(fi) > 0
    assert "feature" in fi.columns
    assert "importance" in fi.columns


def test_feature_importance_all_features_present(trained_model):
    model, _ = trained_model
    fi = model.get_feature_importance()
    for feat in XG_BASIC_FEATURES:
        assert feat in fi["feature"].values


def test_model_handles_missing_features(trained_model):
    """Modelo deve funcionar mesmo com features faltando (preenche com 0)."""
    model, df = trained_model
    X_partial = df[["distance_to_goal", "angle_to_goal"]]
    preds = model.predict(X_partial)
    assert len(preds) == len(df)


def test_model_save_and_load(trained_model, tmp_path):
    model, df = trained_model
    path = str(tmp_path / "basic_xg.pkl")
    model.save(path)
    assert os.path.exists(path)
    loaded = BasicXGModel.load(path)
    assert loaded.is_trained is True
    original_preds = model.predict(df[XG_BASIC_FEATURES])
    loaded_preds = loaded.predict(df[XG_BASIC_FEATURES])
    np.testing.assert_allclose(original_preds, loaded_preds, atol=1e-6)


def test_train_result_string_representation(sample_shots_df):
    model = BasicXGModel()
    X = sample_shots_df[XG_BASIC_FEATURES]
    y = sample_shots_df["is_goal"]
    result = model.train(X, y)
    s = str(result)
    assert "BasicXGModel" in s
    assert "AUC" in s
