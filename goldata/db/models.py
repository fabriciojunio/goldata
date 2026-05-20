"""Modelos SQLAlchemy ORM do GolData."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ── Match ─────────────────────────────────────────────────────────────────────

class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    competition: Mapped[str] = mapped_column(String(100), nullable=False)
    season: Mapped[str] = mapped_column(String(20), nullable=False)
    home_team: Mapped[str] = mapped_column(String(100), nullable=False)
    away_team: Mapped[str] = mapped_column(String(100), nullable=False)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    match_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    stadium: Mapped[str | None] = mapped_column(String(200), nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    shots: Mapped[list["Shot"]] = relationship("Shot", back_populates="match", lazy="select")
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="match", lazy="select"
    )

    __table_args__ = (
        Index("ix_matches_competition_season", "competition", "season"),
        Index("ix_matches_teams", "home_team", "away_team"),
    )


# ── Team ──────────────────────────────────────────────────────────────────────

class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    league: Mapped[str] = mapped_column(String(100), nullable=False)
    season: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    __table_args__ = (Index("ix_teams_league_season", "league", "season"),)


# ── Player (LGPD: nome hasheado) ──────────────────────────────────────────────

class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="SHA-256 do nome real (LGPD)"
    )
    display_name: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="Nome público como conhecido no futebol"
    )
    position: Mapped[str | None] = mapped_column(String(50), nullable=True)
    team_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=True, index=True
    )
    birth_year: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Apenas ano (LGPD)"
    )
    nationality: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    stats: Mapped[list["PlayerStats"]] = relationship(
        "PlayerStats", back_populates="player", lazy="select"
    )

    __table_args__ = (Index("ix_players_name_hash", "name_hash"),)


# ── Shot ──────────────────────────────────────────────────────────────────────

class Shot(Base):
    __tablename__ = "shots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=True, index=True
    )
    player_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    team_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    # Posição
    x: Mapped[float | None] = mapped_column(Float, nullable=True)
    y: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    distance: Mapped[float | None] = mapped_column(Float, nullable=True)
    angle: Mapped[float | None] = mapped_column(Float, nullable=True)

    # xG
    xg_basic: Mapped[float | None] = mapped_column(Float, nullable=True)
    xg_advanced: Mapped[float | None] = mapped_column(Float, nullable=True)
    xg_positional: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Características
    body_part: Mapped[str | None] = mapped_column(String(50), nullable=True)
    technique: Mapped[str | None] = mapped_column(String(100), nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_goal: Mapped[bool] = mapped_column(Boolean, default=False)
    minute: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    match: Mapped["Match | None"] = relationship("Match", back_populates="shots")

    __table_args__ = (Index("ix_shots_match_player", "match_id", "player_id"),)


# ── PlayerStats ───────────────────────────────────────────────────────────────

class PlayerStats(Base):
    __tablename__ = "player_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("players.id"), nullable=False, index=True
    )
    season: Mapped[str] = mapped_column(String(20), nullable=False)
    competition: Mapped[str] = mapped_column(String(100), nullable=False)
    minutes_played: Mapped[float] = mapped_column(Float, default=0)

    # Ofensivo
    goals: Mapped[float] = mapped_column(Float, default=0)
    assists: Mapped[float] = mapped_column(Float, default=0)
    xg: Mapped[float] = mapped_column(Float, default=0)
    xa: Mapped[float] = mapped_column(Float, default=0)
    shots: Mapped[float] = mapped_column(Float, default=0)
    shots_on_target: Mapped[float] = mapped_column(Float, default=0)

    # Passe
    passes: Mapped[float] = mapped_column(Float, default=0)
    passes_completed: Mapped[float] = mapped_column(Float, default=0)
    key_passes: Mapped[float] = mapped_column(Float, default=0)
    progressive_passes: Mapped[float] = mapped_column(Float, default=0)

    # Defensivo
    tackles: Mapped[float] = mapped_column(Float, default=0)
    interceptions: Mapped[float] = mapped_column(Float, default=0)
    blocks: Mapped[float] = mapped_column(Float, default=0)
    pressures: Mapped[float] = mapped_column(Float, default=0)

    # Posse
    touches: Mapped[float] = mapped_column(Float, default=0)
    dribbles: Mapped[float] = mapped_column(Float, default=0)
    dribbles_completed: Mapped[float] = mapped_column(Float, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    player: Mapped["Player"] = relationship("Player", back_populates="stats")

    __table_args__ = (
        Index("ix_player_stats_season_comp", "player_id", "season", "competition"),
    )


# ── TeamStats ─────────────────────────────────────────────────────────────────

class TeamStats(Base):
    __tablename__ = "team_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    season: Mapped[str] = mapped_column(String(20), nullable=False)
    competition: Mapped[str] = mapped_column(String(100), nullable=False)
    matches_played: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    draws: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    points: Mapped[int] = mapped_column(Integer, default=0)
    goals_for: Mapped[float] = mapped_column(Float, default=0)
    goals_against: Mapped[float] = mapped_column(Float, default=0)
    xg_for: Mapped[float] = mapped_column(Float, default=0)
    xg_against: Mapped[float] = mapped_column(Float, default=0)
    xpts: Mapped[float] = mapped_column(Float, default=0)
    ppda: Mapped[float | None] = mapped_column(Float, nullable=True)
    possession_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (Index("ix_team_stats_season_comp", "team_id", "season", "competition"),)


# ── Prediction ────────────────────────────────────────────────────────────────

class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=True, index=True
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    home_win_prob: Mapped[float] = mapped_column(Float, nullable=False)
    draw_prob: Mapped[float] = mapped_column(Float, nullable=False)
    away_win_prob: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_home_goals: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_away_goals: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    match: Mapped["Match | None"] = relationship("Match", back_populates="predictions")


# ── ScoutingReport ────────────────────────────────────────────────────────────

class ScoutingReport(Base):
    __tablename__ = "scouting_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    similarity_cluster: Mapped[int | None] = mapped_column(Integer, nullable=True)
    market_value_estimated: Mapped[float | None] = mapped_column(Float, nullable=True)
    scouting_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    report_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# ── AuditLog (LGPD) ───────────────────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False, index=True
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_id_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    user_identifier_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    purpose: Mapped[str] = mapped_column(String(500), nullable=False)
    details_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
