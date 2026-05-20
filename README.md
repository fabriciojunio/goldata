# GolData ⚽

Plataforma completa de analytics de futebol com Machine Learning. Calcula Expected Goals (xG), analisa redes de passe, prediz resultados e identifica valor em apostas.

## Funcionalidades

- **xG (Expected Goals)**: Modelo XGBoost com features de posição, ângulo e pressão
- **Expected Assists (xA)**: Predição de probabilidade de assistência por passe
- **Rede de Passes**: Análise com NetworkX — centralidade, hubs, fluidez tática
- **Predição de Resultados**: Modelos Poisson, Dixon-Coles, Elo e Monte Carlo
- **Scouting**: Clustering de jogadores similares e projeção de mercado
- **Cartola FC**: Predição de pontuação para fantasy football brasileiro
- **API REST**: FastAPI com autenticação JWT, rate limiting e documentação Swagger

## Stack

- **Python 3.11+** (linguagem principal)
- **FastAPI** (API REST)
- **XGBoost + scikit-learn** (modelos ML)
- **NetworkX** (análise de redes de passe)
- **SQLAlchemy + Alembic** (banco de dados)
- **Plotly + Seaborn** (visualizações)
- **Docker** (containerização)

## Como Executar

```bash
# Com Docker (recomendado)
docker compose up -d

# Ou localmente
pip install -e ".[dev]"
uvicorn goldata.api.main:app --reload --port 8000
```

## Documentação da API

Acesse `http://localhost:8000/docs` para a documentação interativa Swagger.

## Licença

MIT
