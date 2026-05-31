"""Testes do banco de dados SQLAlchemy async."""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

from goldata.db.models import (
    Base, Match, Team, Player, Shot, PlayerStats,
    TeamStats, Prediction, ScoutingReport, AuditLog,
)
from goldata.db.connection import check_connection, create_all_tables, drop_all_tables
from goldata.security import hash_personal_data
from datetime import datetime, timezone


# Engine de teste em memória (SQLite)
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_engine():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine):
    session_maker = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_maker() as session:
        yield session


# ── ORM Models ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_match(db_session):
    match = Match(
        competition="Brasileirão Série A",
        season="2024",
        home_team="Flamengo",
        away_team="Palmeiras",
        home_score=2,
        away_score=1,
    )
    db_session.add(match)
    await db_session.commit()
    await db_session.refresh(match)
    assert match.id is not None
    assert match.home_team == "Flamengo"


@pytest.mark.asyncio
async def test_create_team(db_session):
    team = Team(name="Flamengo", country="Brasil", league="Brasileirão", season="2024")
    db_session.add(team)
    await db_session.commit()
    await db_session.refresh(team)
    assert team.id is not None
    assert team.name == "Flamengo"


@pytest.mark.asyncio
async def test_create_player_lgpd(db_session):
    """Player deve armazenar nome como hash (LGPD)."""
    name_hash = hash_personal_data("Neymar Jr")
    player = Player(
        name_hash=name_hash,
        display_name="Neymar",
        position="FW",
        birth_year=1992,
        nationality="Brasileiro",
    )
    db_session.add(player)
    await db_session.commit()
    await db_session.refresh(player)
    assert player.id is not None
    assert player.name_hash == name_hash
    assert "Neymar Jr" not in player.name_hash  # nome real não armazenado


@pytest.mark.asyncio
async def test_create_shot(db_session):
    shot = Shot(
        x=108.0, y=40.0, distance=12.0, angle=0.6,
        xg_basic=0.35, xg_advanced=0.38, xg_positional=0.32,
        body_part="right_foot", is_goal=True, minute=67,
    )
    db_session.add(shot)
    await db_session.commit()
    await db_session.refresh(shot)
    assert shot.id is not None
    assert shot.is_goal is True


@pytest.mark.asyncio
async def test_create_player_stats(db_session):
    player = Player(name_hash=hash_personal_data("Test Player"), display_name="Test")
    db_session.add(player)
    await db_session.flush()

    stats = PlayerStats(
        player_id=player.id,
        season="2024", competition="Brasileirão",
        minutes_played=2700, goals=15, assists=8,
        xg=14.2, xa=7.8,
    )
    db_session.add(stats)
    await db_session.commit()
    await db_session.refresh(stats)
    assert stats.id is not None
    assert stats.goals == 15


@pytest.mark.asyncio
async def test_create_team_stats(db_session):
    ts = TeamStats(
        team_id=1, season="2024", competition="Brasileirão",
        matches_played=38, wins=25, draws=8, losses=5,
        points=83, goals_for=72, goals_against=35,
        xg_for=68.5, xg_against=32.1, xpts=78.3,
    )
    db_session.add(ts)
    await db_session.commit()
    await db_session.refresh(ts)
    assert ts.id is not None
    assert ts.points == 83


@pytest.mark.asyncio
async def test_create_prediction(db_session):
    pred = Prediction(
        model_name="DixonColes",
        model_version="1.0.0",
        home_win_prob=0.55,
        draw_prob=0.25,
        away_win_prob=0.20,
        predicted_home_goals=1.8,
        predicted_away_goals=1.1,
        confidence=0.75,
    )
    db_session.add(pred)
    await db_session.commit()
    await db_session.refresh(pred)
    total = pred.home_win_prob + pred.draw_prob + pred.away_win_prob
    assert abs(total - 1.0) < 0.01


@pytest.mark.asyncio
async def test_create_scouting_report(db_session):
    report = ScoutingReport(
        player_id=1,
        similarity_cluster=3,
        market_value_estimated=15.5,
        scouting_score=0.82,
        report_json={"strengths": ["pace", "finishing"], "weaknesses": ["aerial"]},
    )
    db_session.add(report)
    await db_session.commit()
    await db_session.refresh(report)
    assert report.id is not None
    assert report.report_json["strengths"][0] == "pace"


@pytest.mark.asyncio
async def test_create_audit_log_lgpd(db_session):
    """AuditLog deve registrar acesso a dados pessoais (LGPD)."""
    log = AuditLog(
        action="read",
        entity_type="player",
        entity_id_hash=hash_personal_data("player_123"),
        user_identifier_hash=hash_personal_data("analyst_456"),
        purpose="analytics esportivo",
    )
    db_session.add(log)
    await db_session.commit()
    await db_session.refresh(log)
    assert log.id is not None
    assert log.action == "read"


@pytest.mark.asyncio
async def test_match_relationship_with_shots(db_session):
    """Match deve ter relação com Shots."""
    match = Match(
        competition="Brasileirão", season="2024",
        home_team="Flamengo", away_team="Vasco",
        home_score=3, away_score=0,
    )
    db_session.add(match)
    await db_session.flush()

    shot1 = Shot(match_id=match.id, x=110.0, y=40.0, is_goal=True)
    shot2 = Shot(match_id=match.id, x=95.0, y=35.0, is_goal=False)
    db_session.add_all([shot1, shot2])
    await db_session.commit()

    from sqlalchemy import select
    result = await db_session.execute(select(Shot).where(Shot.match_id == match.id))
    shots = result.scalars().all()
    assert len(shots) == 2


@pytest.mark.asyncio
async def test_db_query_select_1(db_session):
    """Verificar que SELECT 1 funciona."""
    result = await db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1


@pytest.mark.asyncio
async def test_check_connection():
    """check_connection deve retornar bool."""
    result = await check_connection()
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_create_and_drop_tables():
    """create_all_tables e drop_all_tables devem funcionar."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    
    # Substituir engine temporariamente
    import goldata.db.connection as conn_module
    original_engine = conn_module.engine
    conn_module.engine = engine
    
    try:
        await create_all_tables()
        await drop_all_tables()
    finally:
        conn_module.engine = original_engine
        await engine.dispose()


@pytest.mark.asyncio
async def test_player_birth_year_only(db_session):
    """Apenas ano de nascimento é armazenado (LGPD: minimização)."""
    player = Player(
        name_hash=hash_personal_data("Some Player"),
        display_name="Some Player",
        birth_year=1995,  # apenas o ano, não data completa
    )
    db_session.add(player)
    await db_session.commit()
    await db_session.refresh(player)
    assert player.birth_year == 1995


@pytest.mark.asyncio  
async def test_multiple_predictions_for_match(db_session):
    """Uma partida pode ter previsões de múltiplos modelos."""
    match = Match(competition="Brasileirão", season="2024",
                  home_team="São Paulo", away_team="Corinthians")
    db_session.add(match)
    await db_session.flush()

    for model in ["Elo", "Poisson", "DixonColes"]:
        pred = Prediction(
            match_id=match.id, model_name=model, model_version="1.0.0",
            home_win_prob=0.45, draw_prob=0.30, away_win_prob=0.25,
        )
        db_session.add(pred)
    
    await db_session.commit()
    
    from sqlalchemy import select
    result = await db_session.execute(
        select(Prediction).where(Prediction.match_id == match.id)
    )
    preds = result.scalars().all()
    assert len(preds) == 3
