# GolData ⚽

Plataforma completa de analytics de futebol com Machine Learning. Calcula Expected Goals (xG), analisa redes de passe, prediz resultados e identifica valor em apostas.

## Funcionalidades

- **xG (Expected Goals)**: Modelo XGBoost com features de posição, ângulo e pressão
- **Expected Assists (xA)**: Predição de probabilidade de assistência por passe
- **Rede de Passes**: Análise com NetworkX: centralidade, hubs, fluidez tática
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

## Como rodar localmente

### Pré-requisitos

- **Python 3.11+**
- **Docker** e **Docker Compose** (recomendado)

### Com Docker (recomendado)

```bash
# 1. Clone o repositório
git clone https://github.com/fabriciojunio/goldata.git
cd goldata

# 2. Configure as variáveis de ambiente
cp .env.example .env

# 3. Suba os serviços
docker compose up -d
```

Acesse:
- **API:** [http://localhost:8000](http://localhost:8000)
- **Documentação Swagger:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Sem Docker (ambiente local)

```bash
# 1. Clone o repositório
git clone https://github.com/fabriciojunio/goldata.git
cd goldata

# 2. Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# 3. Instale as dependências
pip install -e ".[dev]"

# 4. Configure as variáveis de ambiente
cp .env.example .env

# 5. Inicie a API
uvicorn goldata.api.main:app --reload --port 8000
```

## Documentação da API

Acesse [http://localhost:8000/docs](http://localhost:8000/docs) para a documentação interativa Swagger.

## Licença

MIT
