# GolData ⚽📊

**Football Analytics Platform with Machine Learning**

Motor de analytics esportivos open-source da [Ictus Technologies](https://ictustech.com.br).
Modelos de xG, previsão de partidas, scouting, análise tática e betting intelligence para o futebol brasileiro e global.

---

## Destaques

| Módulo | Tecnologia | Descrição |
|---|---|---|
| **xG Models** | XGBoost + LightGBM + Bayesian | Expected Goals com SHAP explainability |
| **Match Prediction** | Dixon-Coles + Elo + Poisson | Previsão de resultados com walk-forward validation |
| **Player Scouting** | K-Means + Cosine Similarity | Clustering de perfis e motor de similaridade |
| **Tactical Analysis** | NetworkX + PPDA | Redes de passe e análise de pressing |
| **Betting Engine** | Kelly Criterion + Backtest | Value bets com gestão de banca |
| **xT Metrics** | Expected Threat Grid | Valorização de ações no campo |
| **LGPD Compliance** | Fernet + SHA-256 | Anonimização e direitos do titular |
| **REST API** | FastAPI + Pydantic | Endpoints documentados com rate limiting |

---

## Instalação rápida

```bash
git clone https://github.com/ictustech/goldata.git
cd goldata
pip install -e ".[dev]"
```

### Rodar a API

```bash
uvicorn goldata.api.main:app --reload
# Docs: http://localhost:8000/docs
```

### Rodar os testes

```bash
pytest --cov=goldata --cov-report=term-missing
# 308 testes | 85% cobertura
```

---

## Uso rápido

### xG (Expected Goals)

```python
from goldata.models.xg.advanced import AdvancedXGModel
from goldata.data.features import FeatureEngineer

fe = FeatureEngineer()
model = AdvancedXGModel()
model.train(X_train, y_train)

shot = {"x": 108.0, "y": 40.0, "is_penalty": 0, "body_part": "right foot"}
features = fe.extract_shot_features(shot)
xg = model.predict(pd.DataFrame([features]))[0]
print(f"xG: {xg:.3f}")  # ex: 0.312
```

### Previsão de Partida (Dixon-Coles)

```python
from goldata.models.prediction.dixon_coles import DixonColes

model = DixonColes()
model.fit(results_df)  # DataFrame com home_team, away_team, home_goals, away_goals

pred = model.predict_match("Flamengo", "Palmeiras")
print(f"Flamengo: {pred['home_win_prob']:.1%} | Empate: {pred['draw_prob']:.1%} | Palmeiras: {pred['away_win_prob']:.1%}")
```

### Simulação Monte Carlo do Brasileirão

```python
from goldata.models.prediction.monte_carlo import LeagueSimulator

sim = LeagueSimulator(prediction_model=model)
result = sim.simulate(current_table, remaining_fixtures, n_simulations=10_000)

libertadores = sim.get_libertadores_race(result)
print(libertadores.head(4))  # G4 — Libertadores
```

### Scouting — Jogadores Similares

```python
from goldata.models.scouting.similarity import PlayerSimilarityEngine

engine = PlayerSimilarityEngine()
engine.fit(player_stats_df)

similares = engine.find_similar("player_neymar", n=5, same_position=True)
for p in similares:
    print(f"{p.display_name} — Similaridade: {p.similarity_score:.3f}")
```

### Value Bets

```python
from goldata.models.betting.value_detector import ValueBetDetector
from goldata.models.betting.kelly import kelly_stake

detector = ValueBetDetector(min_edge=0.03)
bets = detector.detect(
    "Flamengo", "Palmeiras",
    model_probs={"home_win": 0.62, "draw": 0.22, "away_win": 0.16},
    market_odds={"home_win": 2.10, "draw": 3.40, "away_win": 5.00},
)
for bet in bets:
    stake = kelly_stake(bankroll=1000.0, prob=bet.model_prob, odd=bet.odd)
    print(f"💰 {bet.market}: edge={bet.edge:.1%} | EV={bet.expected_value:.3f} | stake=R${stake['stake_amount']:.2f}")
```

---

## Arquitetura

```
goldata/
├── goldata/
│   ├── config.py              # Settings singleton (pydantic-settings)
│   ├── security.py            # Fernet + LGPD compliance
│   ├── cache.py               # Cache em memória com TTL
│   ├── exceptions.py          # Hierarquia de exceções
│   ├── db/
│   │   ├── connection.py      # SQLAlchemy async + SQLite fallback
│   │   └── models.py          # ORM: Match, Player, Shot, Prediction, AuditLog
│   ├── data/
│   │   ├── features.py        # FeatureEngineer: distância, ângulo, per_90
│   │   ├── validators.py      # Validação de inputs
│   │   └── brasileirao.py     # BrasileiraoDataClient
│   ├── models/
│   │   ├── base.py            # BaseMLModel + TrainResult
│   │   ├── xg/                # BasicXG, AdvancedXG (XGB+LGB+SHAP), Positional
│   │   ├── prediction/        # Elo, Poisson, Dixon-Coles, Monte Carlo
│   │   ├── scouting/          # Clustering, Similarity, Valuation, Projection
│   │   ├── tactical/          # PassingNetwork, Pressing (PPDA)
│   │   └── betting/           # ValueDetector, KellyCriterion, Backtest
│   ├── metrics/
│   │   └── xmetrics.py        # xT, EvaluationMetrics, Calibration
│   └── api/
│       └── main.py            # FastAPI: /xg, /prediction, /betting, /lgpd
└── tests/
    ├── unit/                  # 290+ testes unitários
    └── integration/           # 14 testes de integração
```

---

## Modelos de xG

| Modelo | Algoritmo | Features | AUC (médio) |
|---|---|---|---|
| BasicXG | LR + Random Forest | 8 features básicas | ~0.76 |
| AdvancedXG | XGBoost + LightGBM | 11 features + SHAP | ~0.80 |
| PositionalXG | Bayesian Zone Model | x, y | ~0.72 |

---

## API Endpoints

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/health` | Status da API |
| POST | `/api/v1/xg` | Calcular xG de um chute |
| POST | `/api/v1/prediction` | Prever resultado de partida |
| POST | `/api/v1/betting/value` | Detectar value bets |
| POST | `/api/v1/betting/kelly` | Kelly Criterion para stake |
| POST | `/api/v1/xt` | Expected Threat de posição |
| GET | `/api/v1/lgpd/info` | Informações LGPD |
| GET | `/api/v1/lgpd/export/{id}` | Exportar dados do usuário |

Autenticação: header `X-API-Key`.

---

## LGPD Compliance

O GolData foi desenvolvido em conformidade com a Lei Geral de Proteção de Dados (Lei 13.709/2018):

- **Anonimização**: nomes de jogadores armazenados como SHA-256 hash
- **Minimização**: apenas dados necessários para analytics
- **Retenção**: máximo 365 dias por padrão
- **Direitos**: endpoints para acesso, portabilidade e eliminação (Art. 18)
- **Auditoria**: todo acesso a dados pessoais registrado em `AuditLog`

---

## Roadmap

- [ ] Dashboard React (6 páginas)
- [ ] Integração StatsBomb Open Data
- [ ] Modelo de lesão (injury risk)
- [ ] Cartola FC optimizer
- [ ] Docker Compose com PostgreSQL
- [ ] MLflow tracking
- [ ] Deploy AWS/GCP

### Video Integration (Roadmap)

GolData exporta timestamps de eventos (`match_id, minute, second, event_type`) compatíveis com plataformas de análise de vídeo como **Hudl Sportscode** e **LongoMatch** para tagueamento automático de cenas.

---

## Contribuindo

```bash
git checkout -b feature/minha-feature
pytest  # todos os testes devem passar
ruff check goldata/
git push origin feature/minha-feature
```

---

## Referências Acadêmicas

- Dixon, M.J. e Coles, S.G. (1997). *Modelling Association Football Scores*. Applied Statistics.
- Karun Singh (2019). *Introducing Expected Threat (xT)*. karun.io/blog/expected-threat.
- Dendir, S. (2016). *When do soccer players peak?* Journal of Sports Analytics.

---

## Licença

MIT License — © 2025 Ictus Technologies | Fabrício Júnio | fabricio@ictustech.com.br
