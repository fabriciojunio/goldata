"""GolData FastAPI: Motor de Football Analytics."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from goldata.config import get_settings
from goldata.logging_config import configure_logging, get_logger
from goldata.security import validate_api_key
from goldata.data.features import FeatureEngineer
from goldata.models.xg.basic import BasicXGModel, XG_BASIC_FEATURES
from goldata.models.xg.positional import PositionalXGModel
from goldata.models.prediction.elo import EloRating
from goldata.models.prediction.poisson import BivariatePoisson
from goldata.models.betting.value_detector import ValueBetDetector, odd_to_implied_prob
from goldata.models.betting.kelly import kelly_stake
from goldata.metrics.xmetrics import ExpectedThreat

settings = get_settings()
logger = get_logger(__name__)
fe = FeatureEngineer()
xt_engine = ExpectedThreat()

limiter = Limiter(key_func=get_remote_address)

_xg_model: BasicXGModel | None = None
_elo_model: EloRating | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown do servidor."""
    configure_logging(settings.log_level)
    logger.info("goldata_api_starting", version=settings.version)
    yield
    logger.info("goldata_api_stopping")


app = FastAPI(
    title="GolData API",
    description="Football Analytics Platform: Ictus Technologies",
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Auth dependency ────────────────────────────────────────────────────────────

def verify_api_key(request: Request) -> str:
    api_key = request.headers.get("X-API-Key", "")
    if not validate_api_key(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida",
        )
    return api_key


# ── Schemas ────────────────────────────────────────────────────────────────────

class ShotInput(BaseModel):
    x: float = Field(..., ge=0, le=120, description="Posição x no campo (0-120)")
    y: float = Field(..., ge=0, le=80, description="Posição y no campo (0-80)")
    is_header: int = Field(0, ge=0, le=1)
    is_foot_right: int = Field(1, ge=0, le=1)
    is_foot_left: int = Field(0, ge=0, le=1)
    is_penalty: int = Field(0, ge=0, le=1)
    is_direct_freekick: int = Field(0, ge=0, le=1)
    is_open_play: int = Field(1, ge=0, le=1)


class XGResponse(BaseModel):
    xg: float
    distance_to_goal: float
    angle_to_goal: float
    model: str


class MatchPredictionInput(BaseModel):
    home_team: str
    away_team: str
    competition: str = "brasileirao"


class MatchPredictionResponse(BaseModel):
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    model: str


class ValueBetInput(BaseModel):
    home_team: str
    away_team: str
    model_probs: dict[str, float]
    market_odds: dict[str, float]
    min_edge: float = 0.03


class KellyInput(BaseModel):
    bankroll: float = Field(..., gt=0)
    prob: float = Field(..., gt=0, lt=1)
    odd: float = Field(..., gt=1)
    kelly_fraction: float = Field(0.25, gt=0, le=1)


class XTInput(BaseModel):
    x: float = Field(..., ge=0, le=120)
    y: float = Field(..., ge=0, le=80)


# ── Rotas ──────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
async def health_check() -> dict[str, Any]:
    """Verifica se a API está funcionando."""
    return {
        "status": "healthy",
        "version": settings.version,
        "environment": settings.environment,
        "project": "GolData: Ictus Technologies",
    }


@app.post("/api/v1/xg", response_model=XGResponse, tags=["xG"])
@limiter.limit("60/minute")
async def calculate_xg(
    request: Request,
    shot: ShotInput,
    _: str = Depends(verify_api_key),
) -> XGResponse:
    """Calcula Expected Goals (xG) para um chute."""
    distance = fe.calculate_distance_to_goal(shot.x, shot.y)
    angle = fe.calculate_angle_to_goal(shot.x, shot.y)

    # Usar modelo posicional pré-treinado como fallback rápido
    xg_positional = xt_engine.get_xt_value(shot.x, shot.y) * 3.0
    xg_positional = min(max(xg_positional, 0.01), 0.99)

    if shot.is_penalty:
        xg_positional = 0.76
    elif shot.is_direct_freekick:
        xg_positional = max(xg_positional * 0.7, 0.05)

    return XGResponse(
        xg=round(xg_positional, 4),
        distance_to_goal=round(distance, 2),
        angle_to_goal=round(angle, 4),
        model="positional_v1",
    )


@app.post("/api/v1/prediction", response_model=MatchPredictionResponse, tags=["Previsão"])
@limiter.limit("30/minute")
async def predict_match(
    request: Request,
    match: MatchPredictionInput,
    _: str = Depends(verify_api_key),
) -> MatchPredictionResponse:
    """Prevê resultado de uma partida usando Elo Rating."""
    global _elo_model
    if _elo_model is None:
        _elo_model = EloRating()

    pred = _elo_model.predict_match(match.home_team, match.away_team)
    return MatchPredictionResponse(
        home_team=match.home_team,
        away_team=match.away_team,
        home_win_prob=pred["home_win_prob"],
        draw_prob=pred["draw_prob"],
        away_win_prob=pred["away_win_prob"],
        model="elo_v1",
    )


@app.post("/api/v1/betting/value", tags=["Betting"])
@limiter.limit("30/minute")
async def detect_value_bets(
    request: Request,
    data: ValueBetInput,
    _: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Detecta apostas com valor positivo (value bets)."""
    detector = ValueBetDetector(min_edge=data.min_edge)
    bets = detector.detect(
        data.home_team, data.away_team,
        data.model_probs, data.market_odds,
    )
    return {
        "match": f"{data.home_team} vs {data.away_team}",
        "n_value_bets": len(bets),
        "value_bets": [
            {
                "market": b.market,
                "model_prob": b.model_prob,
                "implied_prob": b.implied_prob,
                "odd": b.odd,
                "edge": b.edge,
                "ev": b.expected_value,
                "stars": b.confidence_stars,
            }
            for b in bets
        ],
    }


@app.post("/api/v1/betting/kelly", tags=["Betting"])
@limiter.limit("60/minute")
async def calculate_kelly(
    request: Request,
    data: KellyInput,
    _: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Calcula tamanho de aposta usando Kelly Criterion fracionado."""
    result = kelly_stake(
        data.bankroll, data.prob, data.odd, data.kelly_fraction
    )
    return result


@app.post("/api/v1/xt", tags=["Métricas"])
@limiter.limit("60/minute")
async def calculate_xt(
    request: Request,
    pos: XTInput,
    _: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Calcula o Expected Threat (xT) de uma posição."""
    xt_value = xt_engine.get_xt_value(pos.x, pos.y)
    return {
        "x": pos.x,
        "y": pos.y,
        "xt": round(xt_value, 6),
        "zone": "attacking_third" if pos.x > 80 else "middle_third" if pos.x > 40 else "defensive_third",
    }


@app.get("/api/v1/lgpd/info", tags=["LGPD"])
async def lgpd_info() -> dict[str, Any]:
    """Informações sobre tratamento de dados pessoais (LGPD Art. 9)."""
    return {
        "controller": "Ictus Technologies",
        "contact": "dpo@ictustech.com.br",
        "purposes": ["analytics esportivo", "previsão de partidas"],
        "legal_basis": "Art. 7, I - consentimento do titular",
        "retention_days": settings.data_retention_days,
        "rights": [
            "Acesso (Art. 18, I)",
            "Correção (Art. 18, III)",
            "Eliminação (Art. 18, VI)",
            "Portabilidade (Art. 18, V)",
        ],
        "dpa_contact": "ANPD - anpd.gov.br",
    }


@app.get("/api/v1/lgpd/export/{user_id}", tags=["LGPD"])
async def export_user_data_endpoint(
    user_id: str,
    _: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Exporta dados do usuário (Art. 18, V - LGPD)."""
    from goldata.security import export_user_data
    return export_user_data(user_id)
