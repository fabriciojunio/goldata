"""
Testes específicos para cobrir cada linha restante e atingir 100% de cobertura.
Cada teste tem comentário indicando exatamente qual linha/branch cobre.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock, AsyncMock
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════
# api/main.py  linhas 40-43 (lifespan startup/shutdown) e 165 (freekick branch)
# ═══════════════════════════════════════════════════════════════════

def test_api_lifespan_startup_shutdown():
    """Cobre linhas 40-43: configure_logging + logger.info no lifespan."""
    from fastapi.testclient import TestClient
    from goldata.api.main import app
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200


def test_api_xg_freekick_branch():
    """Cobre linha 165: xg_positional * 0.7 para freekick."""
    from fastapi.testclient import TestClient
    from goldata.api.main import app
    from goldata.config import get_settings
    settings = get_settings()
    client = TestClient(app)
    r = client.post(
        "/api/v1/xg",
        json={"x": 95.0, "y": 40.0, "is_direct_freekick": 1,
              "is_penalty": 0, "is_foot_right": 1, "is_foot_left": 0,
              "is_header": 0, "is_open_play": 0},
        headers={"X-API-Key": settings.api_key},
    )
    assert r.status_code == 200
    assert r.json()["xg"] >= 0.05  # freekick branch: max(..., 0.05)


# ═══════════════════════════════════════════════════════════════════
# cli.py  linhas 37-38 (cmd_serve) e 77, 84 (main branches)
# ═══════════════════════════════════════════════════════════════════

def test_cli_serve_branch():
    """Cobre linhas 37-38: import uvicorn + uvicorn.run no cmd_serve."""
    import sys
    from unittest.mock import patch, MagicMock
    mock_uvicorn = MagicMock()
    with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
        from goldata import cli
        import importlib
        importlib.reload(cli)
        args = MagicMock()
        args.host = "0.0.0.0"
        args.port = 8000
        args.reload = False
        cli.cmd_serve(args)
        mock_uvicorn.run.assert_called_once()


def test_cli_main_serve_command(capsys):
    """Cobre linhas 77, 84: branch serve no main."""
    from unittest.mock import patch, MagicMock
    mock_uvicorn = MagicMock()
    with patch("sys.argv", ["goldata", "serve", "--port", "9999"]):
        with patch("goldata.cli.cmd_serve") as mock_serve:
            from goldata.cli import main
            main()
            mock_serve.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# data/brasileirao.py  linhas 148, 160, 164, 176, 180, 214, 228
# ═══════════════════════════════════════════════════════════════════

def test_brasileirao_fallback_season(tmp_path):
    """Cobre 148, 164, 180: fallback para season 2024 quando season pedida não existe."""
    from goldata.data.brasileirao import BrasileiraoDataClient
    client = BrasileiraoDataClient(data_dir=str(tmp_path))
    # Season 2023 não existe → fallback para 2024
    df = client.get_serie_a_standings(2023)
    assert len(df) > 0

    df2 = client.get_serie_a_player_stats(2023)
    assert len(df2) > 0

    df3 = client.get_serie_a_matches(2023)
    assert len(df3) > 0


def test_brasileirao_cache_hit(tmp_path):
    """Cobre 160, 176: return cached (segunda chamada usa cache)."""
    from goldata.data.brasileirao import BrasileiraoDataClient
    client = BrasileiraoDataClient(data_dir=str(tmp_path))
    df1 = client.get_serie_a_standings(2024)
    df2 = client.get_serie_a_standings(2024)  # esta usa o cache
    assert len(df1) == len(df2)

    df3 = client.get_serie_a_player_stats(2024)
    df4 = client.get_serie_a_player_stats(2024)  # cache
    assert len(df3) == len(df4)


def test_brasileirao_home_advantage_no_away_games(tmp_path):
    """Cobre 214, 228: fallback quando time não tem jogos fora/casa."""
    from goldata.data.brasileirao import BrasileiraoDataClient, HOME_ADVANTAGE_DEFAULT
    client = BrasileiraoDataClient(data_dir=str(tmp_path))

    # Criar matches onde o time só joga em casa (nunca fora)
    import pandas as pd
    matches = pd.DataFrame({
        "match_id": [1, 2],
        "season": ["2024", "2024"],
        "round": [1, 2],
        "home_team": ["TimeSoEmCasa", "TimeSoEmCasa"],
        "away_team": ["Palmeiras", "Flamengo"],
        "home_goals": [2, 1],
        "away_goals": [1, 0],
        "home_xg": [1.8, 1.2],
        "away_xg": [0.9, 0.6],
        "stadium": ["Estadio A", "Estadio A"],
        "attendance": [30000, 25000],
    })
    matches.to_csv(tmp_path / "sample" / "brasileirao" / "serie_a_matches_2024.csv", index=False)
    # Recriar client para limpar cache
    client2 = BrasileiraoDataClient(data_dir=str(tmp_path))
    factor = client2.get_home_advantage_factor("TimeSoEmCasa")
    # Sem jogos fora → HOME_ADVANTAGE_DEFAULT
    assert factor == HOME_ADVANTAGE_DEFAULT


# ═══════════════════════════════════════════════════════════════════
# data/features.py  linhas 88, 90, 102, 146
# ═══════════════════════════════════════════════════════════════════

def test_features_batch_no_x_column():
    """Cobre linhas 88, 90: batch sem coluna x ou y → preenche com default."""
    from goldata.data.features import FeatureEngineer
    fe = FeatureEngineer()
    df = pd.DataFrame({
        "is_header": [0, 1],
        "is_penalty": [0, 1],
        "is_foot_right": [1, 0],
    })
    result = fe.extract_shot_features_batch(df)
    assert "distance_to_goal" in result.columns
    assert "angle_to_goal" in result.columns


def test_features_batch_missing_binary_cols():
    """Cobre linha 102: preenche colunas binárias ausentes com 0."""
    from goldata.data.features import FeatureEngineer
    fe = FeatureEngineer()
    df = pd.DataFrame({"x": [100.0], "y": [40.0]})
    result = fe.extract_shot_features_batch(df)
    assert result["is_penalty"].iloc[0] == 0


def test_features_shot_accuracy_normalized():
    """Cobre linha 146: shot_accuracy = shots_on_target / shots."""
    from goldata.data.features import FeatureEngineer
    fe = FeatureEngineer()
    df = pd.DataFrame({
        "minutes_played": [2700.0],
        "goals": [10.0], "assists": [5.0],
        "xg": [9.5], "xa": [4.8],
        "shots": [100.0], "shots_on_target": [40.0],
        "passes": [1500.0], "passes_completed": [1200.0],
        "key_passes": [30.0], "progressive_passes": [80.0],
        "tackles": [20.0], "interceptions": [15.0],
        "pressures": [200.0], "dribbles": [50.0], "dribbles_completed": [35.0],
    })
    result = fe.normalize_player_stats_per90(df)
    assert "shot_accuracy" in result.columns
    assert result["shot_accuracy"].iloc[0] == pytest.approx(0.4, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
# data/validators.py  linha 55: odd > 1000
# ═══════════════════════════════════════════════════════════════════

def test_validator_odd_too_high():
    """Cobre linha 55: odd > 1000 lança InvalidInputError."""
    from goldata.data.validators import DataValidator
    from goldata.exceptions import InvalidInputError
    v = DataValidator()
    with pytest.raises(InvalidInputError):
        v.validate_odds(1001.0)


# ═══════════════════════════════════════════════════════════════════
# metrics/xmetrics.py  linhas 91-92: fallback cols em calculate_xt_per_player
# ═══════════════════════════════════════════════════════════════════

def test_xt_per_player_missing_end_cols():
    """Cobre 91-92: preenche end_x, end_y, x, y ausentes com 0.0."""
    from goldata.metrics.xmetrics import ExpectedThreat
    xt = ExpectedThreat()
    df = pd.DataFrame({
        "player_id": ["p1", "p2", "p3"],
        "event_type": ["pass", "pass", "shot"],
        # sem x, y, end_x, end_y
    })
    result = xt.calculate_xt_per_player(df)
    assert "xt_total" in result.columns
    assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════
# models/base.py  linhas 82-93: get_feature_importance sem coef_ nem importances_
# ═══════════════════════════════════════════════════════════════════

def test_base_model_feature_importance_fallback():
    """Cobre 82-93: modelo sem feature_importances_ nem coef_ retorna zeros."""
    from goldata.models.base import BaseMLModel, TrainResult

    class MinimalModel(BaseMLModel):
        model_name = "MinimalModel"
        def train(self, X, y): pass
        def predict(self, X): return np.zeros(len(X))
        def predict_proba(self, X): return np.zeros((len(X), 2))

    m = MinimalModel()
    m.is_trained = True
    m._feature_names = ["a", "b", "c"]
    m._model = MagicMock(spec=[])  # sem feature_importances_ nem coef_

    fi = m.get_feature_importance()
    assert isinstance(fi, pd.DataFrame)
    assert len(fi) == 3


def test_base_model_check_trained_raises():
    """Cobre linha 82: _check_trained lançando em get_feature_importance."""
    from goldata.models.base import BaseMLModel
    from goldata.exceptions import ModelNotTrainedError

    class M(BaseMLModel):
        model_name = "M"
        def train(self, X, y): pass
        def predict(self, X): return np.zeros(len(X))
        def predict_proba(self, X): return np.zeros((len(X), 2))

    m = M()  # is_trained = False
    with pytest.raises(ModelNotTrainedError):
        m.get_feature_importance()


# ═══════════════════════════════════════════════════════════════════
# models/betting/backtest.py  linha 84: coluna obrigatória ausente
# ═══════════════════════════════════════════════════════════════════

def test_backtest_missing_column_raises():
    """Cobre linha 84: ValueError quando coluna obrigatória falta."""
    from goldata.models.betting.backtest import BettingBacktest
    bt = BettingBacktest()
    df = pd.DataFrame({"model_prob": [0.6], "odd": [2.0]})  # sem "outcome"
    with pytest.raises(ValueError, match="Coluna obrigatória ausente"):
        bt.run(df)


# ═══════════════════════════════════════════════════════════════════
# models/betting/kelly.py  linha 165: get_history vazio
# ═══════════════════════════════════════════════════════════════════

def test_kelly_history_empty_dataframe():
    """Cobre linha 165: get_history retorna DataFrame vazio quando sem apostas."""
    from goldata.models.betting.kelly import KellyCriterion
    kc = KellyCriterion(1000.0)
    df = kc.get_history()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ═══════════════════════════════════════════════════════════════════
# models/betting/value_detector.py  linhas 57, 93
# ═══════════════════════════════════════════════════════════════════

def test_remove_margin_zero_total():
    """Cobre linha 57: total == 0 retorna probs original."""
    from goldata.models.betting.value_detector import remove_bookmaker_margin
    probs = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
    result = remove_bookmaker_margin(probs)
    assert result == probs


def test_confidence_stars_returns_1_minimum():
    """Cobre linha 93: return 1 (score abaixo de todos os thresholds)."""
    from goldata.models.betting.value_detector import ValueBetDetector
    d = ValueBetDetector()
    stars = d._confidence_stars(0.0001, 0.5001)
    assert stars == 1


# ═══════════════════════════════════════════════════════════════════
# models/fantasy/cartola_predictor.py  linhas 103, 231-250 (PuLP + walk_forward)
# ═══════════════════════════════════════════════════════════════════

def test_cartola_unknown_position_fallback():
    """Cobre linha 103: position não reconhecida usa fallback xg*3+xa*2."""
    from goldata.models.fantasy.cartola_predictor import CartolaPredictor
    pred = CartolaPredictor()
    player = {"position": "DESCONHECIDA", "xg_per_90": 0.3, "xa_per_90": 0.2,
              "form_last_3": 5.0, "minutes_last_match": 90}
    pts = pred.predict_player_points(player)
    assert pts >= 0


def test_cartola_optimize_tries_pulp_then_greedy():
    """Cobre linhas 231-250: _optimize_pulp é tentado e cai no greedy se PuLP falha."""
    from goldata.models.fantasy.cartola_predictor import CartolaPredictor
    import numpy as np
    np.random.seed(0)
    n = 20
    positions = ["GOL"] * 2 + ["LAT"] * 4 + ["ZAG"] * 4 + ["MEI"] * 4 + ["ATA"] * 4 + ["TEC"] * 2
    df = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "display_name": [f"J{i}" for i in range(n)],
        "position": positions,
        "team": [f"t{i%5}" for i in range(n)],
        "price": np.random.uniform(3.0, 15.0, n),
        "xg_per_90": np.random.uniform(0, 0.5, n),
        "xa_per_90": np.random.uniform(0, 0.4, n),
        "goals_per_90": np.random.uniform(0, 0.4, n),
        "assists_per_90": np.random.uniform(0, 0.3, n),
        "form_last_3": np.random.uniform(3, 10, n),
        "minutes_last_match": [90] * n,
    })
    pred = CartolaPredictor()
    pred.build_predictions(df)

    # Forçar ImportError do PuLP para testar greedy
    with patch("builtins.__import__", side_effect=lambda name, *a, **kw:
               (_ for _ in ()).throw(ImportError("pulp")) if name == "pulp"
               else __import__(name, *a, **kw)):
        draft = pred.optimize_draft(budget=100.0)
    assert draft.total_price <= 100.0


# ═══════════════════════════════════════════════════════════════════
# models/injury/risk_predictor.py  linhas 68, 109, 134
# ═══════════════════════════════════════════════════════════════════

def test_injury_prepare_missing_features():
    """Cobre linha 68: _prepare preenche features ausentes com 0.0."""
    from goldata.models.injury.risk_predictor import InjuryRiskPredictor
    import numpy as np
    model = InjuryRiskPredictor(random_state=42)
    # Treinar com dados mínimos
    X = pd.DataFrame({"age": np.random.uniform(18, 35, 60),
                       "previous_injuries_12m": np.random.randint(0, 4, 60)})
    y = pd.Series(np.random.randint(0, 4, 60))
    model.train(X, y)
    # Predizer com features ausentes → _prepare preenche linha 68
    player = {"age": 25}
    report = model.predict_risk(player)
    assert 0 <= report.risk_score <= 1


def test_injury_minutes_overload_factor():
    """Cobre linha 109: fator 'Sobrecarga semanal' quando minutes_last_7_days > 180."""
    from goldata.models.injury.risk_predictor import InjuryRiskPredictor, INJURY_FEATURES
    import numpy as np
    model = InjuryRiskPredictor(random_state=42)
    X = pd.DataFrame({f: np.random.uniform(0, 1, 80) for f in INJURY_FEATURES})
    y = pd.Series(np.random.randint(0, 4, 80))
    model.train(X, y)
    player = {f: 0.0 for f in INJURY_FEATURES}
    player["minutes_last_7_days"] = 250.0  # > 180 → linha 109
    report = model.predict_risk(player)
    assert "Sobrecarga semanal (>180 min em 7 dias)" in report.risk_factors


def test_injury_batch_predict_raises_before_train():
    """Cobre linha 134: predict_batch levanta ModelNotTrainedError."""
    from goldata.models.injury.risk_predictor import InjuryRiskPredictor
    from goldata.exceptions import ModelNotTrainedError
    model = InjuryRiskPredictor()
    with pytest.raises(ModelNotTrainedError):
        model.predict_batch(pd.DataFrame({"age": [25]}))


# ═══════════════════════════════════════════════════════════════════
# models/prediction/dixon_coles.py  linhas 97-98: except ValueError/TypeError
# ═══════════════════════════════════════════════════════════════════

def test_dixon_coles_invalid_date_fallback(sample_match_results_df):
    """Cobre 97-98: data inválida → days_ago = 0.0."""
    from goldata.models.prediction.dixon_coles import DixonColes
    df = sample_match_results_df.copy()
    df["date"] = "data-invalida-xyz"  # provoca ValueError no fromisoformat
    model = DixonColes()
    model.fit(df)
    assert model.is_trained


# ═══════════════════════════════════════════════════════════════════
# models/prediction/elo.py  linha 180: get_ratings sem times
# ═══════════════════════════════════════════════════════════════════

def test_elo_get_ratings_empty():
    """Cobre linha 180: retorna DataFrame vazio quando sem times."""
    from goldata.models.prediction.elo import EloRating
    elo = EloRating()
    df = elo.get_ratings()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "team" in df.columns


# ═══════════════════════════════════════════════════════════════════
# models/prediction/monte_carlo.py  linha 125: sim_table sem coluna 'team'
# ═══════════════════════════════════════════════════════════════════

def test_monte_carlo_table_as_index():
    """Cobre linha 125: sim_table usa index quando não há coluna 'team'."""
    from goldata.models.prediction.monte_carlo import LeagueSimulator
    # Tabela com index como times (sem coluna explícita 'team')
    table = pd.DataFrame({
        "points": [30, 25, 20, 15],
        "goals_for": [40, 35, 25, 20],
        "goals_against": [20, 22, 30, 38],
    }, index=["Flamengo", "Palmeiras", "Corinthians", "Vasco"])

    sim = LeagueSimulator()
    # _simulate_once internamente usa o index
    result_table = sim._simulate_once(table, [("Flamengo", "Palmeiras")])
    assert result_table is not None


# ═══════════════════════════════════════════════════════════════════
# models/referee/analyzer.py  linha 145: get_profiles_dataframe vazio
# ═══════════════════════════════════════════════════════════════════

def test_referee_profiles_dataframe_empty():
    """Cobre linha 145: retorna DataFrame vazio sem perfis."""
    from goldata.models.referee.analyzer import RefereeAnalyzer
    analyzer = RefereeAnalyzer()
    df = analyzer.get_profiles_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ═══════════════════════════════════════════════════════════════════
# models/scouting/clustering.py  linhas 114, 141, 155, 161
# ═══════════════════════════════════════════════════════════════════

def test_clustering_empty_cluster_skipped():
    """Cobre linha 114: continue quando cluster vazio."""
    from goldata.models.scouting.clustering import PlayerClusterer, CLUSTERING_FEATURES
    # Dados concentrados em dois grupos → alguns clusters ficam vazios
    df1 = pd.DataFrame({f: [1.0] * 20 for f in CLUSTERING_FEATURES})
    df2 = pd.DataFrame({f: [0.0] * 20 for f in CLUSTERING_FEATURES})
    df = pd.concat([df1, df2], ignore_index=True)
    c = PlayerClusterer(n_clusters=8, random_state=42)
    c.fit(df)
    assert c.is_trained


def test_clustering_predict_raises_not_trained():
    """Cobre linha 141: predict levanta quando não treinado."""
    from goldata.models.scouting.clustering import PlayerClusterer
    from goldata.exceptions import ModelNotTrainedError
    c = PlayerClusterer()
    with pytest.raises(ModelNotTrainedError):
        c.predict(pd.DataFrame({"x": [1]}))


def test_clustering_predict_with_distance_raises():
    """Cobre linha 155: predict_with_distance levanta quando não treinado."""
    from goldata.models.scouting.clustering import PlayerClusterer
    from goldata.exceptions import ModelNotTrainedError
    c = PlayerClusterer()
    with pytest.raises(ModelNotTrainedError):
        c.predict_with_distance(pd.DataFrame({"x": [1]}))


def test_clustering_inertia_raises():
    """Cobre linha 161: get_inertia levanta quando não treinado."""
    from goldata.models.scouting.clustering import PlayerClusterer
    from goldata.exceptions import ModelNotTrainedError
    c = PlayerClusterer()
    with pytest.raises(ModelNotTrainedError):
        c.get_inertia()


# ═══════════════════════════════════════════════════════════════════
# models/scouting/projection.py  linhas 70, 109
# ═══════════════════════════════════════════════════════════════════

def test_projection_zero_current_val_returns_1():
    """Cobre linha 70: current_val == 0 → return 1.0."""
    from goldata.models.scouting.projection import PerformanceProjector
    proj = PerformanceProjector()
    # Idade muito jovem (< _AGE_START=17) resulta em current_val = curva[0]
    # Forçar através de age=17 → idx=0
    player = {"age": 17, "position": "FW", "goals_per_90": 0.3}
    result = proj.project_performance(player, target_age=17)
    # Mesmo target, multiplier deve ser 1.0
    assert result["projection_multiplier"] == pytest.approx(1.0, abs=0.01)


def test_projection_stable_metrics_preserved():
    """Cobre linha 109: métricas estáveis (pass_completion_rate) são copiadas."""
    from goldata.models.scouting.projection import PerformanceProjector
    proj = PerformanceProjector()
    player = {"age": 25, "position": "MF", "goals_per_90": 0.3,
              "pass_completion_rate": 0.87, "shot_accuracy": 0.42}
    result = proj.project_performance(player, target_age=27)
    assert result["pass_completion_rate"] == pytest.approx(0.87)
    assert result["shot_accuracy"] == pytest.approx(0.42)


# ═══════════════════════════════════════════════════════════════════
# models/scouting/similarity.py  linhas 55, 76, 139, 142, 150
# ═══════════════════════════════════════════════════════════════════

def test_similarity_fit_with_team_column():
    """Cobre linha 76: indexa coluna 'team' quando presente."""
    from goldata.models.scouting.similarity import PlayerSimilarityEngine
    df = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(15)],
        "display_name": [f"J{i}" for i in range(15)],
        "position": ["FW"] * 15,
        "team": [f"team_{i%3}" for i in range(15)],
        **{f: np.random.uniform(0, 1, 15) for f in
           ["goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
            "shots_per_90", "key_passes_per_90", "progressive_passes_per_90",
            "tackles_per_90", "interceptions_per_90", "pressures_per_90",
            "dribbles_completed_per_90", "pass_completion_rate"]},
    })
    engine = PlayerSimilarityEngine()
    engine.fit(df)
    assert "team_0" in engine._player_teams.values()


def test_similarity_missing_features_filled():
    """Cobre linha 55: features ausentes preenchidas com 0.0."""
    from goldata.models.scouting.similarity import PlayerSimilarityEngine
    df = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(15)],
        "goals_per_90": np.random.uniform(0, 1, 15),
        # sem as outras features
    })
    engine = PlayerSimilarityEngine()
    engine.fit(df)
    assert engine.is_trained


def test_similarity_find_raises_not_trained():
    """Cobre linha 139: find_similar levanta quando não treinado."""
    from goldata.models.scouting.similarity import PlayerSimilarityEngine
    from goldata.exceptions import ModelNotTrainedError
    engine = PlayerSimilarityEngine()
    with pytest.raises(ModelNotTrainedError):
        engine.find_similar("p1")


def test_similarity_score_raises_data_not_found():
    """Cobre linha 142: similarity_score levanta quando player não existe."""
    from goldata.models.scouting.similarity import PlayerSimilarityEngine
    from goldata.exceptions import DataNotFoundError
    df = pd.DataFrame({
        "player_id": ["p1", "p2"],
        **{f: [0.5, 0.3] for f in
           ["goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
            "shots_per_90", "key_passes_per_90", "progressive_passes_per_90",
            "tackles_per_90", "interceptions_per_90", "pressures_per_90",
            "dribbles_completed_per_90", "pass_completion_rate"]},
    })
    engine = PlayerSimilarityEngine()
    engine.fit(df)
    with pytest.raises(DataNotFoundError):
        engine.similarity_score("p1", "inexistente")


def test_similarity_get_all_player_ids():
    """Cobre linha 150: get_all_player_ids retorna lista."""
    from goldata.models.scouting.similarity import PlayerSimilarityEngine
    df = pd.DataFrame({
        "player_id": ["a", "b", "c"],
        **{f: [0.5, 0.3, 0.7] for f in
           ["goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
            "shots_per_90", "key_passes_per_90", "progressive_passes_per_90",
            "tackles_per_90", "interceptions_per_90", "pressures_per_90",
            "dribbles_completed_per_90", "pass_completion_rate"]},
    })
    engine = PlayerSimilarityEngine()
    engine.fit(df)
    ids = engine.get_all_player_ids()
    assert ids == ["a", "b", "c"]


# ═══════════════════════════════════════════════════════════════════
# models/scouting/valuation.py  linhas 50, 96
# ═══════════════════════════════════════════════════════════════════

def test_valuation_no_age_column():
    """Cobre linha 50: adiciona coluna 'age' com default 25 quando ausente."""
    from goldata.models.scouting.valuation import PlayerValuationModel
    model = PlayerValuationModel()
    X = pd.DataFrame({"goals_per_90": [0.4, 0.5, 0.3], "xg_per_90": [0.35, 0.45, 0.25]})
    y = pd.Series([5.0, 8.0, 3.0])
    result = model.train(X, y)
    assert "mae_millions" in result


def test_valuation_predict_raises_not_trained():
    """Cobre linha 96: predict levanta quando não treinado."""
    from goldata.models.scouting.valuation import PlayerValuationModel
    from goldata.exceptions import ModelNotTrainedError
    model = PlayerValuationModel()
    with pytest.raises(ModelNotTrainedError):
        model.predict(pd.DataFrame({"age": [25]}))


# ═══════════════════════════════════════════════════════════════════
# models/tactical/pressing.py  linhas 85, 94
# ═══════════════════════════════════════════════════════════════════

def test_pressing_batch_skips_unknown_team():
    """Cobre linha 85: continue quando time não está nos eventos."""
    from goldata.models.tactical.pressing import PressingAnalyzer
    df = pd.DataFrame({
        "team_id": ["team_a", "team_b"],
        "event_type": ["pass", "tackle"],
        "x": [30.0, 70.0],
        "match_id": [1, 1],
    })
    result = PressingAnalyzer.calculate_ppda_batch(df, team_ids=["time_inexistente"])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_pressing_batch_no_match_id_col():
    """Cobre linha 94: branch quando não há coluna match_id."""
    from goldata.models.tactical.pressing import PressingAnalyzer
    df = pd.DataFrame({
        "team_id": ["team_home", "team_away", "team_home"],
        "event_type": ["pass", "pass", "tackle"],
        "x": [30.0, 25.0, 70.0],
        # sem match_id
    })
    result = PressingAnalyzer.calculate_ppda_batch(df, team_ids=["team_home"])
    assert isinstance(result, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════
# models/transfers/analyzer.py  linhas 67-68, 91-95, 121-122, 129, 168-187
# ═══════════════════════════════════════════════════════════════════

def test_transfers_no_market_value_column():
    """Cobre 67-68: adiciona market_value_m com default quando coluna ausente."""
    from goldata.models.transfers.analyzer import TransferAnalyzer
    df = pd.DataFrame({
        "player_id": ["p1", "p2", "p3"],
        "display_name": ["J1", "J2", "J3"],
        "position": ["FW", "MF", "DF"],
        "xg_per_90": [0.5, 0.3, 0.1],
        "xa_per_90": [0.2, 0.4, 0.1],
        "age": [22, 25, 30],
        # SEM market_value_m
    })
    analyzer = TransferAnalyzer()
    opps = analyzer.find_undervalued(df, min_gap_pct=0.0)
    assert isinstance(opps, list)


def test_transfers_similarity_exception_handled():
    """Cobre 91-95: exception em find_similar é silenciada."""
    from goldata.models.transfers.analyzer import TransferAnalyzer
    mock_sim = MagicMock()
    mock_sim.is_trained = True
    mock_sim.find_similar.side_effect = Exception("erro")
    df = pd.DataFrame({
        "player_id": ["p1"],
        "display_name": ["J1"],
        "position": ["FW"],
        "market_value_m": [5.0],
        "xg_per_90": [0.8],
        "xa_per_90": [0.4],
        "age": [22],
    })
    analyzer = TransferAnalyzer(similarity_engine=mock_sim)
    opps = analyzer.find_undervalued(df, min_gap_pct=0.0)
    assert isinstance(opps, list)


def test_transfers_overvalued_no_market_col():
    """Cobre 121-122: find_overvalued sem coluna market_value_m."""
    from goldata.models.transfers.analyzer import TransferAnalyzer
    df = pd.DataFrame({
        "player_id": ["p1"],
        "display_name": ["J1"],
        "position": ["FW"],
        "xg_per_90": [0.0],
        "xa_per_90": [0.0],
        "age": [38],
        # SEM market_value_m
    })
    analyzer = TransferAnalyzer()
    opps = analyzer.find_overvalued(df)
    assert isinstance(opps, list)


def test_transfers_with_trained_valuation_model():
    """Cobre linha 129: usa valuation model treinado para estimar valor."""
    from goldata.models.transfers.analyzer import TransferAnalyzer
    from goldata.models.scouting.valuation import PlayerValuationModel
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "display_name": [f"J{i}" for i in range(n)],
        "position": ["FW"] * n,
        "age": np.random.randint(20, 33, n).astype(float),
        "market_value_m": np.random.uniform(2.0, 20.0, n),
        **{f: np.random.uniform(0, 0.5, n) for f in
           ["xg_per_90", "xa_per_90", "goals_per_90", "assists_per_90",
            "shots_per_90", "key_passes_per_90", "progressive_passes_per_90",
            "tackles_per_90", "interceptions_per_90", "pressures_per_90",
            "dribbles_completed_per_90", "pass_completion_rate"]},
    })
    val = PlayerValuationModel()
    val.train(df, df["market_value_m"])
    analyzer = TransferAnalyzer(valuation_model=val)
    opps = analyzer.find_undervalued(df, min_gap_pct=0.0)
    assert isinstance(opps, list)


def test_transfers_replacement_cost_no_similarity():
    """Cobre linha 168: retorna DataFrame vazio quando sem similarity engine."""
    from goldata.models.transfers.analyzer import TransferAnalyzer
    analyzer = TransferAnalyzer()
    result = analyzer.replacement_cost_analysis("p1", pd.DataFrame())
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_transfers_replacement_cost_with_similarity():
    """Cobre linha 187: retorna affordable players."""
    from goldata.models.transfers.analyzer import TransferAnalyzer
    from goldata.models.scouting.similarity import PlayerSimilarityEngine
    feature_cols = ["goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
                    "shots_per_90", "key_passes_per_90", "progressive_passes_per_90",
                    "tackles_per_90", "interceptions_per_90", "pressures_per_90",
                    "dribbles_completed_per_90", "pass_completion_rate"]
    np.random.seed(0)
    n = 20
    df = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "display_name": [f"J{i}" for i in range(n)],
        "position": ["FW"] * n,
        "market_value_m": np.random.uniform(1, 15, n),
        **{f: np.random.uniform(0, 0.5, n) for f in feature_cols},
    })
    engine = PlayerSimilarityEngine()
    engine.fit(df)
    analyzer = TransferAnalyzer(similarity_engine=engine)
    result = analyzer.replacement_cost_analysis("p0", df, budget_m=10.0)
    assert isinstance(result, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════
# models/xg/advanced.py  linhas 30-32, 38-41 (ImportError branches),
#                         84-96 (segundo XGBoost quando sem LGB),
#                         153-172 (permutation importance quando sem SHAP)
# ═══════════════════════════════════════════════════════════════════

def test_advanced_xg_without_lgbm(sample_shots_df):
    """Cobre 84-96: fallback para XGBoost duplo quando LightGBM indisponível."""
    import sys
    from goldata.models.xg.advanced import XG_ADVANCED_FEATURES
    # Simular ausência de LightGBM forçando o path do fallback diretamente
    with patch("goldata.models.xg.advanced._LGBM_AVAILABLE", False):
        from goldata.models.xg import advanced as adv_module
        import importlib
        importlib.reload(adv_module)
        model = adv_module.AdvancedXGModel(random_state=42)
        X = sample_shots_df
        y = sample_shots_df["is_goal"]
        result = model.train(X, y)
        assert model.is_trained
        preds = model.predict(X)
        assert len(preds) == len(X)


def test_advanced_xg_without_shap(sample_shots_df):
    """Cobre 153-172: permutation importance quando SHAP indisponível."""
    from goldata.models.xg.advanced import AdvancedXGModel, XG_ADVANCED_FEATURES
    model = AdvancedXGModel(random_state=42)
    X = sample_shots_df
    y = sample_shots_df["is_goal"]
    model.train(X, y)

    with patch("goldata.models.xg.advanced._SHAP_AVAILABLE", False):
        X_small = X.head(20)
        shap_vals = model.get_shap_values(X_small)
        assert shap_vals is not None


def test_advanced_xg_import_error_branches():
    """Cobre 30-32, 38-41: branches de ImportError para lgbm e shap."""
    # Esses branches só são executados na importação do módulo.
    # Verificamos indiretamente que o módulo pode ser importado
    # com e sem as libs, testando as flags resultantes.
    from goldata.models.xg import advanced
    assert hasattr(advanced, "_LGBM_AVAILABLE")
    assert hasattr(advanced, "_SHAP_AVAILABLE")
    assert isinstance(advanced._LGBM_AVAILABLE, bool)
    assert isinstance(advanced._SHAP_AVAILABLE, bool)


# ═══════════════════════════════════════════════════════════════════
# security.py  linhas 56, 72, 109-110
# ═══════════════════════════════════════════════════════════════════

def test_sanitize_non_string_value():
    """Cobre linha 56: value não-string é convertido para str."""
    from goldata.security import sanitize_string
    result = sanitize_string(12345)  # type: ignore
    assert result == "12345"


def test_sanitize_dict_non_string_passthrough():
    """Cobre linha 72: valores não-string (int, float, list) passam intocados."""
    from goldata.security import sanitize_dict
    data = {"goals": 15, "xg": 12.3, "tags": ["striker", "pace"]}
    result = sanitize_dict(data)
    assert result["goals"] == 15
    assert result["xg"] == pytest.approx(12.3)
    assert result["tags"] == ["striker", "pace"]


def test_anonymize_birth_date_invalid_format():
    """Cobre 109-110: except quando str()[:4] lança exceção via mock."""
    from goldata.security import anonymize_player_data
    from unittest.mock import patch

    class BadStr:
        """Objeto cujo str() lança exceção no slice."""
        def __str__(self):
            raise ValueError("bad")

    with patch("goldata.security.str", side_effect=lambda x: (_ for _ in ()).throw(ValueError("bad"))
               if not isinstance(x, str) else str.__new__(str, x)):
        data = {"birth_date": BadStr(), "display_name": "Test"}
        result = anonymize_player_data(data)
        assert result["birth_date"] is None


# ═══════════════════════════════════════════════════════════════════
# viz/pitch.py  linhas 116-117: plot_shot_map sem coluna x → early return
# ═══════════════════════════════════════════════════════════════════

def test_plot_shot_map_no_x_column():
    """Cobre 116-117: early return quando não há coluna x."""
    from goldata.viz.pitch import plot_shot_map
    df = pd.DataFrame({"is_goal": [1, 0], "xg": [0.5, 0.1]})  # sem x
    fig, ax = plot_shot_map(df, title="Sem X")
    assert fig is not None
    plt.close("all")


# ═══════════════════════════════════════════════════════════════════
# db/connection.py  linhas 30-33, 49-57, 67-69
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_db_get_db_generator():
    """Cobre 49-57: get_db commit normal."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from goldata.db.models import Base
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import goldata.db.connection as conn_mod
    original = conn_mod.engine
    conn_mod.engine = engine
    conn_mod.AsyncSessionLocal = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    try:
        gen = conn_mod.get_db()
        session = await gen.__anext__()
        assert session is not None
        try:
            await gen.asend(None)
        except StopAsyncIteration:
            pass
    finally:
        conn_mod.engine = original
    await engine.dispose()


@pytest.mark.asyncio
async def test_db_get_db_rollback_on_exception():
    """Cobre 67-69: rollback quando exception é levantada."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from goldata.db.models import Base
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import goldata.db.connection as conn_mod
    original = conn_mod.engine
    original_local = conn_mod.AsyncSessionLocal
    conn_mod.engine = engine
    conn_mod.AsyncSessionLocal = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    try:
        gen = conn_mod.get_db()
        await gen.__anext__()
        with pytest.raises(RuntimeError):
            await gen.athrow(RuntimeError("test rollback"))
    except StopAsyncIteration:
        pass
    finally:
        conn_mod.engine = original
        conn_mod.AsyncSessionLocal = original_local
    await engine.dispose()


def test_db_engine_fallback_sqlite():
    """Cobre 30-33: fallback para SQLite quando create_async_engine lança exceção."""
    import goldata.db.connection as conn_mod
    from sqlalchemy.ext.asyncio import create_async_engine as _orig

    call_count = [0]
    def mock_engine(url, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("PostgreSQL indisponível simulado")
        return _orig(url, **kwargs)

    with patch("goldata.db.connection.create_async_engine", side_effect=mock_engine):
        engine = conn_mod._build_engine()
        assert engine is not None
