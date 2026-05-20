"""Testes do modelo xG posicional."""

import pytest
import numpy as np
import pandas as pd
from goldata.models.xg.positional import PositionalXGModel, GRID_X, GRID_Y
from goldata.exceptions import ModelNotTrainedError


@pytest.fixture
def trained_positional(sample_shots_df):
    model = PositionalXGModel()
    X = sample_shots_df[["x", "y"]]
    y = sample_shots_df["is_goal"]
    result = model.train(X, y)
    return model, sample_shots_df, result


def test_positional_trains_without_error(sample_shots_df):
    model = PositionalXGModel()
    result = model.train(sample_shots_df[["x", "y"]], sample_shots_df["is_goal"])
    assert model.is_trained is True


def test_positional_predict_before_train_raises():
    model = PositionalXGModel()
    with pytest.raises(ModelNotTrainedError):
        model.predict(pd.DataFrame({"x": [100], "y": [40]}))


def test_positional_predict_returns_probabilities(trained_positional):
    model, df, _ = trained_positional
    preds = model.predict(df[["x", "y"]])
    assert (preds >= 0).all()
    assert (preds <= 1).all()


def test_positional_grid_shape(trained_positional):
    model, _, _ = trained_positional
    grid = model.get_xg_grid()
    assert grid.shape == (GRID_X, GRID_Y)


def test_positional_grid_values_in_range(trained_positional):
    model, _, _ = trained_positional
    grid = model.get_xg_grid()
    assert (grid >= 0).all()
    assert (grid <= 1).all()


def test_positional_heatmap_alias(trained_positional):
    model, _, _ = trained_positional
    grid = model.get_xg_grid()
    heatmap = model.get_zone_heatmap()
    np.testing.assert_array_equal(grid, heatmap)


def test_positional_closer_shot_higher_xg():
    """Treina com dados onde chutes perto têm taxa de gol maior."""
    np.random.seed(1)
    n = 600
    X = pd.DataFrame({
        "x": np.concatenate([np.full(300, 115.0), np.full(300, 60.0)]),
        "y": np.full(600, 40.0),
    })
    y = pd.Series(np.concatenate([
        np.random.binomial(1, 0.35, 300),
        np.random.binomial(1, 0.02, 300)
    ]))
    model = PositionalXGModel()
    model.train(X, y)
    close = pd.DataFrame({"x": [115.0], "y": [40.0]})
    far = pd.DataFrame({"x": [60.0], "y": [40.0]})
    assert model.predict(close)[0] > model.predict(far)[0]


def test_positional_xg_for_position(trained_positional):
    model, _, _ = trained_positional
    xg = model.get_xg_for_position(108, 40)
    assert 0 <= xg <= 1


def test_positional_penalty_area_xg(trained_positional):
    """Zona de área deve ter xG razoável (> prior)."""
    model, _, _ = trained_positional
    xg = model.get_xg_for_position(112, 40)
    assert xg > 0.05  # acima do prior mínimo


def test_positional_predict_proba_shape(trained_positional):
    model, df, _ = trained_positional
    proba = model.predict_proba(df[["x", "y"]])
    assert proba.shape == (len(df), 2)


def test_positional_predict_proba_sums_to_1(trained_positional):
    model, df, _ = trained_positional
    proba = model.predict_proba(df[["x", "y"]])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_positional_auc_above_minimum(trained_positional):
    model, df, result = trained_positional
    assert result.train_auc >= 0.50


def test_positional_feature_importance_returns_df(trained_positional):
    model, _, _ = trained_positional
    fi = model.get_feature_importance()
    assert isinstance(fi, pd.DataFrame)
    assert len(fi) == 2
