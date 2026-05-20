FROM python:3.12-slim

WORKDIR /app

# Dependências de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "." 2>/dev/null || pip install --no-cache-dir \
    fastapi uvicorn pydantic pydantic-settings sqlalchemy alembic aiosqlite \
    pandas numpy scipy scikit-learn xgboost lightgbm shap matplotlib networkx \
    requests httpx structlog cryptography joblib slowapi python-multipart

# Copiar código
COPY goldata/ goldata/
COPY data/ data/

# Criar diretório de dados
RUN mkdir -p data/sample/brasileirao data/models data/cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "goldata.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
