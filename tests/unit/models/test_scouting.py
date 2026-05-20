"""Testes dos módulos de scouting: clustering, similaridade, valuation, projeção."""

import pytest
import numpy as np
import pandas as pd

from goldata.models.scouting.clustering import PlayerClusterer, CLUSTERING_FEATURES
from goldata.models.scouting.similarity import PlayerSimilarityEngine
from goldata.models.scouting.valuation import PlayerValuationModel
from goldata.models.scouting.projection import PerformanceProjector, PEAK_WINDOWS
from goldata.exceptions import ModelNotTrainedError, InsufficientDataError, DataNotFoundError
from goldata.data.features import FeatureEngineer


@pytest.fixture
def per90_stats(sample_player_stats_df):
    """Estatísticas por 90 minutos."""
    fe = FeatureEngineer()
    return fe.normalize_player_stats_per90(sample_player_stats_df)


@pytest.fixture
def trained_clusterer(per90_stats):
    c = PlayerClusterer(n_clusters=8, random_state=42)
    c.fit(per90_stats)
    return c, per90_stats


@pytest.fixture
def trained_similarity(per90_stats, sample_player_stats_df):
    df = per90_stats.copy()
    df["player_id"] = sample_player_stats_df["player_id"]
    df["display_name"] = sample_player_stats_df["display_name"]
    df["position"] = sample_player_stats_df["position"]
    engine = PlayerSimilarityEngine()
    engine.fit(df)
    return engine, df


@pytest.fixture
def valuation_data(per90_stats, sample_player_stats_df):
    df = per90_stats.copy()
    df["age"] = np.random.randint(18, 35, len(df))
    np.random.seed(42)
    # Valor de mercado sintético correlacionado com xg e idade
    y = pd.Series(
        np.clip(df.get("xg_per_90", pd.Series(np.zeros(len(df)))) * 15
                + (30 - df["age"]) * 0.5
                + np.random.normal(0, 2, len(df)), 0.5, 50),
        name="market_value_m"
    )
    return df, y


# ── Clustering ────────────────────────────────────────────────────────────────

def test_clusterer_trains(per90_stats):
    c = PlayerClusterer(n_clusters=8, random_state=42)
    c.fit(per90_stats)
    assert c.is_trained is True


def test_clusterer_predict_returns_array(trained_clusterer):
    c, df = trained_clusterer
    labels = c.predict(df)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(df)


def test_clusterer_labels_in_range(trained_clusterer):
    c, df = trained_clusterer
    labels = c.predict(df)
    assert (labels >= 0).all()
    assert (labels < 8).all()


def test_clusterer_predict_with_distance(trained_clusterer):
    c, df = trained_clusterer
    result = c.predict_with_distance(df)
    assert "cluster_id" in result.columns
    assert "cluster_archetype" in result.columns
    assert "distance_to_centroid" in result.columns


def test_clusterer_profiles_built(trained_clusterer):
    c, _ = trained_clusterer
    profiles = c.get_cluster_profiles()
    assert len(profiles) > 0


def test_clusterer_inertia_positive(trained_clusterer):
    c, _ = trained_clusterer
    assert c.get_inertia() > 0


def test_clusterer_raises_before_train():
    c = PlayerClusterer()
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ModelNotTrainedError):
        c.predict(df)


def test_clusterer_raises_insufficient_data():
    c = PlayerClusterer(n_clusters=8)
    df = pd.DataFrame({feat: [1, 2, 3] for feat in CLUSTERING_FEATURES})
    with pytest.raises(InsufficientDataError):
        c.fit(df)


def test_clusterer_handles_missing_features(trained_clusterer):
    c, _ = trained_clusterer
    df_minimal = pd.DataFrame({"goals_per_90": [0.5, 0.3], "assists_per_90": [0.2, 0.1]})
    labels = c.predict(df_minimal)
    assert len(labels) == 2


# ── Similarity ────────────────────────────────────────────────────────────────

def test_similarity_fits(per90_stats, sample_player_stats_df):
    df = per90_stats.copy()
    df["player_id"] = sample_player_stats_df["player_id"]
    df["display_name"] = sample_player_stats_df["display_name"]
    engine = PlayerSimilarityEngine()
    engine.fit(df)
    assert engine.is_trained is True


def test_similarity_find_similar_returns_list(trained_similarity):
    engine, df = trained_similarity
    player_id = df["player_id"].iloc[0]
    results = engine.find_similar(player_id, n=5)
    assert isinstance(results, list)
    assert len(results) <= 5


def test_similarity_scores_in_range(trained_similarity):
    engine, df = trained_similarity
    player_id = df["player_id"].iloc[0]
    results = engine.find_similar(player_id, n=10)
    for r in results:
        assert -1.0 <= r.similarity_score <= 1.0


def test_similarity_excludes_self(trained_similarity):
    engine, df = trained_similarity
    player_id = df["player_id"].iloc[0]
    results = engine.find_similar(player_id, n=10)
    ids = [r.player_id for r in results]
    assert player_id not in ids


def test_similarity_score_between_two_players(trained_similarity):
    engine, df = trained_similarity
    pid_a = df["player_id"].iloc[0]
    pid_b = df["player_id"].iloc[1]
    score = engine.similarity_score(pid_a, pid_b)
    assert -1.0 <= score <= 1.0


def test_similarity_unknown_player_raises(trained_similarity):
    engine, _ = trained_similarity
    with pytest.raises(DataNotFoundError):
        engine.find_similar("jogador_inexistente_xyz")


def test_similarity_raises_before_fit():
    engine = PlayerSimilarityEngine()
    with pytest.raises(ModelNotTrainedError):
        engine.find_similar("qualquer")


def test_similarity_same_position_filter(trained_similarity):
    engine, df = trained_similarity
    pid = df["player_id"].iloc[0]
    results = engine.find_similar(pid, n=5, same_position=True)
    target_pos = engine._player_positions.get(pid)
    if target_pos and results:
        for r in results:
            assert engine._player_positions.get(r.player_id) == target_pos


# ── Valuation ─────────────────────────────────────────────────────────────────

def test_valuation_trains(valuation_data):
    df, y = valuation_data
    model = PlayerValuationModel()
    result = model.train(df, y)
    assert model.is_trained is True
    assert "mae_millions" in result


def test_valuation_predict_positive(valuation_data):
    df, y = valuation_data
    model = PlayerValuationModel()
    model.train(df, y)
    preds = model.predict(df)
    assert (preds >= 0).all()


def test_valuation_predict_single(valuation_data):
    df, y = valuation_data
    model = PlayerValuationModel()
    model.train(df, y)
    player = {"age": 24, "goals_per_90": 0.6, "xg_per_90": 0.5}
    value = model.predict_single(player)
    assert value >= 0


def test_valuation_feature_importance(valuation_data):
    df, y = valuation_data
    model = PlayerValuationModel()
    model.train(df, y)
    fi = model.get_feature_importance()
    assert len(fi) > 0
    assert "feature" in fi.columns


def test_valuation_raises_before_train():
    model = PlayerValuationModel()
    with pytest.raises(ModelNotTrainedError):
        model.predict(pd.DataFrame({"age": [25]}))


# ── Projection ────────────────────────────────────────────────────────────────

def test_projection_get_age_curve():
    proj = PerformanceProjector()
    curve = proj.get_age_curve("FW")
    assert len(curve) == 24  # idades 17-40
    assert max(curve) == pytest.approx(1.0)


def test_projection_peak_window_fw():
    proj = PerformanceProjector()
    peak = proj.get_peak_window("FW")
    assert peak == (25, 29)


def test_projection_peak_window_gk():
    proj = PerformanceProjector()
    peak = proj.get_peak_window("GK")
    assert peak == (28, 34)


def test_projection_multiplier_young_player():
    """Jogador jovem (21) deve crescer até o pico (27 para MF)."""
    proj = PerformanceProjector()
    player = {"age": 21, "position": "MF", "goals_per_90": 0.3}
    projected = proj.project_performance(player, target_age=27)
    assert projected["goals_per_90"] > player["goals_per_90"]


def test_projection_multiplier_declining():
    """Jogador no final da carreira deve ter performance projetada menor."""
    proj = PerformanceProjector()
    player = {"age": 32, "position": "FW", "goals_per_90": 0.4}
    projected = proj.project_performance(player, target_age=36)
    assert projected["goals_per_90"] < player["goals_per_90"]


def test_projection_in_peak_window(proj=None):
    p = PerformanceProjector()
    player = {"age": 23, "position": "FW", "goals_per_90": 0.5}
    proj = p.project_performance(player, target_age=27)
    assert proj["in_peak_window"] is True


def test_projection_career_trajectory():
    proj = PerformanceProjector()
    player = {"age": 22, "position": "MF"}
    traj = proj.get_career_trajectory(player, age_range=(20, 35))
    assert len(traj) == 16
    assert "performance_index" in traj.columns
    assert "in_peak" in traj.columns


def test_projection_unknown_position_fallback():
    proj = PerformanceProjector()
    curve = proj.get_age_curve("UNKNOWN")
    assert len(curve) == 24
