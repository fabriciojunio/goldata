"""Rota de health check da API GolData."""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check():
    """Verifica se a API está funcionando corretamente."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "goldata-api",
        "version": "1.0.0",
    }
