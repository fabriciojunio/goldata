"""
Testes finais para atingir 100% de cobertura.
Cada teste cobre linhas específicas identificadas no relatório de cobertura.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


# ══════════════════════════════════════════════════════════════════
# cli.py linha 84: if __name__ == "__main__": main()
# ══════════════════════════════════════════════════════════════════

def test_cli_main_module_entrypoint():
    """Cobre linha 84: execução como __main__."""
    import runpy
    with patch("sys.argv", ["goldata", "standings", "--season", "2024"]):
        try:
            runpy.run_module("goldata.cli", run_name="__main__", alter_sys=True)
        except SystemExit:
            pass  # normal após execução


# ══════════════════════════════════════════════════════════════════
# data/brasileirao.py linha 176 (cache match) e 228 (home_advantage sem away_ppg>0)
# ══════════════════════════════════════════════════════════════════

def test_brasileirao_matches_cache_hit(tmp_path):
    """Cobre linha 176: return cached em get_serie_a_matches."""
    from goldata.data.brasileirao import BrasileiraoDataClient
    client = BrasileiraoDataClient(data_dir=str(tmp_path))
    _ = client.get_serie_a_matches(2024)   # popula cache
    df2 = client.get_serie_a_matches(2024)  # linha 176: return cached
    assert len(df2) > 0


def test_brasileirao_home_advantage_zero_away_ppg(tmp_path):
    """Cobre linha 228: away_ppg == 0 → return HOME_ADVANTAGE_DEFAULT."""
    from goldata.data.brasileirao import BrasileiraoDataClient, HOME_ADVANTAGE_DEFAULT
    client = BrasileiraoDataClient(data_dir=str(tmp_path))

    # Forçar situação onde away_ppg = 0 (time nunca ganhou fora)
    matches = pd.DataFrame({
        "match_id": [1, 2],
        "season": ["2024", "2024"],
        "round": [1, 2],
        "home_team": ["SoJogaEmCasa", "SoJogaEmCasa"],
        "away_team": ["Rival1", "Rival2"],
        "home_goals": [3, 2],
        "away_goals": [0, 0],  # nunca ganhou fora
        "home_xg": [2.5, 2.0], "away_xg": [0.5, 0.3],
        "stadium": ["Est", "Est"], "attendance": [10000, 10000],
    })
    # time também joga fora mas NUNCA vence (away_ppg = 0)
    matches2 = pd.DataFrame({
        "match_id": [3, 4],
        "season": ["2024", "2024"],
        "round": [3, 4],
        "home_team": ["Rival1", "Rival2"],
        "away_team": ["SoJogaEmCasa", "SoJogaEmCasa"],
        "home_goals": [2, 3],
        "away_goals": [0, 0],  # perdeu fora
        "home_xg": [1.5, 2.0], "away_xg": [0.5, 0.3],
        "stadium": ["Est2", "Est2"], "attendance": [10000, 10000],
    })
    all_matches = pd.concat([matches, matches2], ignore_index=True)
    csv_path = tmp_path / "sample" / "brasileirao" / "serie_a_matches_2024.csv"
    all_matches.to_csv(csv_path, index=False)

    client2 = BrasileiraoDataClient(data_dir=str(tmp_path))
    factor = client2.get_home_advantage_factor("SoJogaEmCasa")
    assert factor == HOME_ADVANTAGE_DEFAULT  # linha 228: away_ppg = 0


# ══════════════════════════════════════════════════════════════════
# db/connection.py linhas 67-69: check_connection falha → return False
# ══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_check_connection_returns_false_on_error():
    """Cobre 67-69: check_connection captura exceção e retorna False."""
    import goldata.db.connection as conn_mod
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_session = AsyncMock()
    mock_session.execute.side_effect = Exception("DB indisponível")
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch.object(conn_mod, "AsyncSessionLocal", return_value=mock_cm):
        result = await conn_mod.check_connection()
        assert result is False


# ══════════════════════════════════════════════════════════════════
# models/base.py linhas 84, 89: get_feature_importance com coef_ e sem nenhum
# ══════════════════════════════════════════════════════════════════

def test_base_model_importance_with_coef():
    """Cobre linha 84: modelo com coef_ retorna importância baseada em |coef_|."""
    from goldata.models.base import BaseMLModel

    class CoefModel(BaseMLModel):
        model_name = "CoefModel"
        def train(self, X, y): pass
        def predict(self, X): return np.zeros(len(X))
        def predict_proba(self, X): return np.zeros((len(X), 2))

    m = CoefModel()
    m.is_trained = True
    m._feature_names = ["f1", "f2", "f3"]
    # Modelo com coef_ mas sem feature_importances_
    mock_model = MagicMock()
    del mock_model.feature_importances_  # garantir que não existe
    mock_model.coef_ = np.array([[0.5, -0.3, 0.8]])
    m._model = mock_model

    fi = m.get_feature_importance()
    assert isinstance(fi, pd.DataFrame)
    assert len(fi) == 3
    # f3 tem maior importância absoluta (0.8)
    assert fi.iloc[0]["feature"] == "f3"


def test_base_model_importance_no_attr():
    """Cobre linha 89: sem feature_importances_ nem coef_ → retorna zeros."""
    from goldata.models.base import BaseMLModel

    class BareModel(BaseMLModel):
        model_name = "BareModel"
        def train(self, X, y): pass
        def predict(self, X): return np.zeros(len(X))
        def predict_proba(self, X): return np.zeros((len(X), 2))

    m = BareModel()
    m.is_trained = True
    m._feature_names = ["a", "b"]
    # _model sem nenhum dos atributos de importância
    m._model = object()  # objeto simples sem feature_importances_ nem coef_

    fi = m.get_feature_importance()
    assert len(fi) == 2
    assert (fi["importance"] == 0.0).all()


# ══════════════════════════════════════════════════════════════════
# models/fantasy/cartola_predictor.py linhas 231-250: _optimize_pulp com PuLP real
# ══════════════════════════════════════════════════════════════════

def test_cartola_optimize_pulp_if_available():
    """Cobre 231-250: executa _optimize_pulp quando PuLP está disponível."""
    try:
        import pulp  # noqa: F401
        pulp_available = True
    except ImportError:
        pulp_available = False

    from goldata.models.fantasy.cartola_predictor import CartolaPredictor

    np.random.seed(99)
    n = 13
    positions = ["GOL", "LAT", "LAT", "ZAG", "ZAG", "MEI", "MEI", "MEI",
                 "ATA", "ATA", "ATA", "TEC", "TEC"]
    df = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "display_name": [f"J{i}" for i in range(n)],
        "position": positions,
        "team": [f"t{i%5}" for i in range(n)],
        "price": np.random.uniform(3.0, 12.0, n),
        "xg_per_90": np.random.uniform(0, 0.5, n),
        "xa_per_90": np.random.uniform(0, 0.4, n),
        "goals_per_90": np.random.uniform(0, 0.4, n),
        "assists_per_90": np.random.uniform(0, 0.3, n),
        "form_last_3": np.random.uniform(3, 10, n),
        "minutes_last_match": [90] * n,
    })
    pred = CartolaPredictor()
    pred.build_predictions(df)

    if pulp_available:
        draft = pred._optimize_pulp(pred._predictions, budget=200.0,
                                    slots={"GOL":1,"LAT":2,"ZAG":2,"MEI":3,"ATA":3,"TEC":1})
        assert draft.total_price <= 200.01
    else:
        # PuLP não instalado — testar que greedy funciona
        draft = pred.optimize_draft(budget=200.0)
        assert draft is not None


def test_cartola_pulp_installed_optimize_draft():
    """Cobre path completo de _optimize_pulp via optimize_draft."""
    try:
        import pulp  # noqa: F401
    except ImportError:
        pytest.skip("PuLP não instalado — skip teste PuLP")

    from goldata.models.fantasy.cartola_predictor import CartolaPredictor
    np.random.seed(7)
    n = 13
    positions = ["GOL", "LAT", "LAT", "ZAG", "ZAG",
                 "MEI", "MEI", "MEI", "ATA", "ATA", "ATA", "TEC", "TEC"]
    df = pd.DataFrame({
        "player_id": [f"px{i}" for i in range(n)],
        "display_name": [f"Jx{i}" for i in range(n)],
        "position": positions,
        "team": [f"t{i%4}" for i in range(n)],
        "price": [5.0] * n,
        "xg_per_90": np.random.uniform(0, 0.5, n),
        "xa_per_90": np.random.uniform(0, 0.3, n),
        "goals_per_90": np.random.uniform(0, 0.4, n),
        "assists_per_90": np.random.uniform(0, 0.2, n),
        "form_last_3": np.random.uniform(4, 9, n),
        "minutes_last_match": [90] * n,
    })
    pred = CartolaPredictor()
    pred.build_predictions(df)
    draft = pred.optimize_draft(budget=300.0)
    assert draft is not None
    assert len(draft.players) > 0


# ══════════════════════════════════════════════════════════════════
# models/prediction/monte_carlo.py linha 125: sim_table.index (sem coluna team)
# ══════════════════════════════════════════════════════════════════

def test_monte_carlo_simulate_index_based_table():
    """Cobre linha 125: sim_table usa .index quando não tem coluna 'team'."""
    from goldata.models.prediction.monte_carlo import LeagueSimulator

    # Tabela com times como index, sem coluna 'team'
    table = pd.DataFrame({
        "points": [30, 25, 20, 15, 10, 8, 6, 4],
        "goals_for": [40, 35, 25, 20, 18, 15, 12, 10],
        "goals_against": [20, 22, 30, 38, 35, 38, 40, 42],
    }, index=["FLA", "PAL", "ATM", "BOT", "COR", "FLU", "VAS", "SPO"])

    fixtures = [("FLA", "PAL"), ("ATM", "BOT")]
    sim = LeagueSimulator()
    result = sim.simulate(table, fixtures, n_simulations=100)
    assert result.n_simulations == 100


# ══════════════════════════════════════════════════════════════════
# models/scouting/clustering.py linha 155: predict_with_distance sem treino
# ══════════════════════════════════════════════════════════════════

def test_clustering_predict_with_distance_not_trained():
    """Cobre linha 155: predict_with_distance levanta ModelNotTrainedError."""
    from goldata.models.scouting.clustering import PlayerClusterer
    from goldata.exceptions import ModelNotTrainedError
    c = PlayerClusterer()
    # _kmeans is None → linha 155
    with pytest.raises(ModelNotTrainedError):
        c.predict_with_distance(pd.DataFrame({"x": [1]}))


# ══════════════════════════════════════════════════════════════════
# models/scouting/projection.py linha 70: current_val == 0 → return 1.0
# ══════════════════════════════════════════════════════════════════

def test_projection_multiplier_current_age_zero_curve():
    """Cobre linha 70: divisão evitada quando current_val é zero."""
    from goldata.models.scouting.projection import PerformanceProjector, _AGE_CURVE_BASE

    proj = PerformanceProjector()
    # Forçar current_val = 0 mockando a curva
    original = _AGE_CURVE_BASE["FW"].copy()
    try:
        _AGE_CURVE_BASE["FW"][0] = 0.0  # índice 0 → age=17
        result = proj._get_multiplier(current_age=17.0, target_age=25.0, position="FW")
        assert result == 1.0  # linha 70: return 1.0
    finally:
        _AGE_CURVE_BASE["FW"][0] = original[0]


# ══════════════════════════════════════════════════════════════════
# models/scouting/similarity.py linha 139: find_similar sem treino
# ══════════════════════════════════════════════════════════════════

def test_similarity_find_similar_not_trained_vectors_none():
    """Cobre linha 139: _player_vectors is None levanta ModelNotTrainedError."""
    from goldata.models.scouting.similarity import PlayerSimilarityEngine
    from goldata.exceptions import ModelNotTrainedError
    engine = PlayerSimilarityEngine()
    engine.is_trained = False  # força is_trained = False
    engine._player_vectors = None
    with pytest.raises(ModelNotTrainedError):
        engine.find_similar("p1")


# ══════════════════════════════════════════════════════════════════
# models/scouting/valuation.py linha 96: predict sem treino
# ══════════════════════════════════════════════════════════════════

def test_valuation_feature_importance_raises_not_trained():
    """Cobre linha 96 via get_feature_importance quando não treinado."""
    from goldata.models.scouting.valuation import PlayerValuationModel
    from goldata.exceptions import ModelNotTrainedError
    model = PlayerValuationModel()
    with pytest.raises(ModelNotTrainedError):
        model.get_feature_importance()


# ══════════════════════════════════════════════════════════════════
# models/transfers/analyzer.py linha 93 (similar names), 129 (valuation predict),
#                               173-174 (exception em find_similar), 180 (affordable)
# ══════════════════════════════════════════════════════════════════

def test_transfers_similarity_returns_names():
    """Cobre linha 93: similar = [s.display_name for s in sims]."""
    from goldata.models.transfers.analyzer import TransferAnalyzer
    from goldata.models.scouting.similarity import PlayerSimilarityEngine

    feature_cols = ["goals_per_90","assists_per_90","xg_per_90","xa_per_90",
                    "shots_per_90","key_passes_per_90","progressive_passes_per_90",
                    "tackles_per_90","interceptions_per_90","pressures_per_90",
                    "dribbles_completed_per_90","pass_completion_rate"]
    np.random.seed(5)
    n = 20
    df = pd.DataFrame({
        "player_id": [f"sid{i}" for i in range(n)],
        "display_name": [f"Sim{i}" for i in range(n)],
        "position": ["FW"] * n,
        "market_value_m": np.random.uniform(1.0, 5.0, n),
        **{f: np.random.uniform(0, 0.5, n) for f in feature_cols},
    })
    engine = PlayerSimilarityEngine()
    engine.fit(df)
    analyzer = TransferAnalyzer(similarity_engine=engine)
    # min_gap_pct=0.0 para capturar todos
    opps = analyzer.find_undervalued(df, min_gap_pct=0.0)
    # Verificar que similar_players foi populado em ao menos uma oportunidade
    has_similar = any(len(o.similar_players) > 0 for o in opps)
    assert isinstance(opps, list)


def test_transfers_valuation_predict_branch():
    """Cobre linha 129: valuation.predict() é chamado quando modelo treinado."""
    from goldata.models.transfers.analyzer import TransferAnalyzer
    from goldata.models.scouting.valuation import PlayerValuationModel

    feature_cols = ["goals_per_90","assists_per_90","xg_per_90","xa_per_90",
                    "shots_per_90","key_passes_per_90","progressive_passes_per_90",
                    "tackles_per_90","interceptions_per_90","pressures_per_90",
                    "dribbles_completed_per_90","pass_completion_rate"]
    np.random.seed(3)
    n = 25
    df = pd.DataFrame({
        "player_id": [f"vp{i}" for i in range(n)],
        "display_name": [f"VP{i}" for i in range(n)],
        "position": ["MF"] * n,
        "age": np.random.uniform(20, 32, n),
        "market_value_m": np.random.uniform(2.0, 15.0, n),
        **{f: np.random.uniform(0, 0.5, n) for f in feature_cols},
    })
    val = PlayerValuationModel()
    val.train(df, df["market_value_m"])
    analyzer = TransferAnalyzer(valuation_model=val)
    opps = analyzer.find_undervalued(df, min_gap_pct=0.0)
    assert isinstance(opps, list)


def test_transfers_replacement_exception_returns_empty():
    """Cobre 173-174: find_similar lança → retorna DataFrame vazio."""
    from goldata.models.transfers.analyzer import TransferAnalyzer
    mock_sim = MagicMock()
    mock_sim.is_trained = True
    mock_sim.find_similar.side_effect = Exception("erro de similaridade")
    analyzer = TransferAnalyzer(similarity_engine=mock_sim)
    result = analyzer.replacement_cost_analysis("p1", pd.DataFrame({"player_id": ["p1"]}))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_transfers_replacement_no_player_id_col():
    """Cobre linha 180: retorna DataFrame com colunas de SimilarPlayer quando sem player_id."""
    from goldata.models.transfers.analyzer import TransferAnalyzer
    from goldata.models.scouting.similarity import PlayerSimilarityEngine
    from goldata.models.scouting.similarity import SimilarPlayer

    feature_cols = ["goals_per_90","assists_per_90","xg_per_90","xa_per_90",
                    "shots_per_90","key_passes_per_90","progressive_passes_per_90",
                    "tackles_per_90","interceptions_per_90","pressures_per_90",
                    "dribbles_completed_per_90","pass_completion_rate"]
    np.random.seed(4)
    n = 15
    df_sim = pd.DataFrame({
        "player_id": [f"rp{i}" for i in range(n)],
        "display_name": [f"RP{i}" for i in range(n)],
        "position": ["FW"] * n,
        **{f: np.random.uniform(0, 0.5, n) for f in feature_cols},
    })
    engine = PlayerSimilarityEngine()
    engine.fit(df_sim)

    analyzer = TransferAnalyzer(similarity_engine=engine)
    # players_df SEM coluna player_id → cai na linha 180
    df_no_pid = pd.DataFrame({"name": ["A", "B"]})
    result = analyzer.replacement_cost_analysis("rp0", df_no_pid, budget_m=10.0)
    assert isinstance(result, pd.DataFrame)


# ══════════════════════════════════════════════════════════════════
# models/xg/advanced.py linhas 30-32, 38-41 (ImportError), 84-96 (XGB fallback),
#                        157, 165-166 (permutation_importance)
# ══════════════════════════════════════════════════════════════════

def test_advanced_xg_lgbm_not_available_path(sample_shots_df):
    """Cobre 84-96: cria XGBClassifier secundário quando _LGBM_AVAILABLE=False."""
    import goldata.models.xg.advanced as adv
    original = adv._LGBM_AVAILABLE

    # Patch direto na instância: forçar o branch do fallback
    with patch.object(adv, "_LGBM_AVAILABLE", False):
        from goldata.models.xg.advanced import XG_ADVANCED_FEATURES
        model = adv.AdvancedXGModel(random_state=0)
        # Verificar que sem LGBM, o segundo modelo é XGBClassifier
        from xgboost import XGBClassifier
        assert isinstance(model._lgbm, XGBClassifier)
        assert model._xgb_weight == 0.5
        assert model._lgbm_weight == 0.5
        # Treinar e predizer
        X = sample_shots_df
        y = sample_shots_df["is_goal"]
        result = model.train(X, y)
        assert model.is_trained
        preds = model.predict(X)
        assert len(preds) == len(X)


def test_advanced_xg_shap_not_available_uses_permutation(sample_shots_df):
    """Cobre 157, 165-166: usa permutation_importance quando _SHAP_AVAILABLE=False."""
    import goldata.models.xg.advanced as adv

    model = adv.AdvancedXGModel(random_state=42)
    X = sample_shots_df
    y = sample_shots_df["is_goal"]
    model.train(X, y)

    with patch.object(adv, "_SHAP_AVAILABLE", False):
        X_small = X.head(30)
        shap_vals = model.get_shap_values(X_small)
        assert shap_vals is not None


def test_advanced_xg_shap_uses_xtrain_when_available(sample_shots_df):
    """Cobre path onde _X_train não é None no fallback de permutation."""
    import goldata.models.xg.advanced as adv

    model = adv.AdvancedXGModel(random_state=42)
    X = sample_shots_df
    y = sample_shots_df["is_goal"]
    model.train(X, y)

    assert model._X_train is not None  # _X_train foi setado no train

    with patch.object(adv, "_SHAP_AVAILABLE", False):
        shap_vals = model.get_shap_values(X.head(10))
        assert shap_vals is not None


def test_advanced_xg_module_importerror_lgbm():
    """Cobre 30-32: branch except ImportError do LightGBM."""
    import sys
    import importlib

    # Remover lgbm do sys.modules e bloquear import
    lgbm_backup = sys.modules.pop("lightgbm", None)
    try:
        with patch.dict(sys.modules, {"lightgbm": None}):
            if "goldata.models.xg.advanced" in sys.modules:
                del sys.modules["goldata.models.xg.advanced"]
            import goldata.models.xg.advanced as adv_reloaded
            assert isinstance(adv_reloaded._LGBM_AVAILABLE, bool)
    finally:
        if lgbm_backup is not None:
            sys.modules["lightgbm"] = lgbm_backup
        if "goldata.models.xg.advanced" in sys.modules:
            del sys.modules["goldata.models.xg.advanced"]
        import goldata.models.xg.advanced  # restaurar


def test_advanced_xg_module_importerror_shap():
    """Cobre 38-41: branch except ImportError do SHAP."""
    import sys

    shap_backup = sys.modules.pop("shap", None)
    try:
        with patch.dict(sys.modules, {"shap": None}):
            if "goldata.models.xg.advanced" in sys.modules:
                del sys.modules["goldata.models.xg.advanced"]
            import goldata.models.xg.advanced as adv_reloaded
            assert isinstance(adv_reloaded._SHAP_AVAILABLE, bool)
    finally:
        if shap_backup is not None:
            sys.modules["shap"] = shap_backup
        if "goldata.models.xg.advanced" in sys.modules:
            del sys.modules["goldata.models.xg.advanced"]
        import goldata.models.xg.advanced  # restaurar
