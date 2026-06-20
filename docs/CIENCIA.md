# Base científica do GolData

Este documento reúne os fundamentos científicos por trás de cada modelo do
motor. Os métodos aqui são o estado-da-prática em analytics de futebol e
apostas — consolidados e ainda atuais. Não substituem julgamento profissional;
desempenho passado não garante resultado futuro.

> Atualização dos **métodos**: consolidados (não envelhecem rápido).
> Atualização dos **dados**: o pacote traz amostra do Brasileirão 2024 — para
> uso real, conecte uma fonte de dados da temporada corrente (ver "Dados").

---

## 1. Previsão de resultados (gols)

### Poisson independente
- **Base**: Maher, M. J. (1982). *Modelling association football scores*. Statistica Neerlandica, 36(3).
- Gols de cada time ~ Poisson com forças de ataque/defesa + vantagem de mando.

### Dixon-Coles
- **Base**: Dixon, M. & Coles, S. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*. JRSS-C, 46(2).
- Corrige a dependência em placares baixos (0-0, 1-0, 0-1, 1-1) e adiciona
  **decaimento temporal** (jogos recentes pesam mais).

### Poisson bivariado (correlação de gols)
- **Base**: Karlis, D. & Ntzoufras, I. (2003). *Analysis of sports data by using bivariate Poisson models*. JRSS-D, 52(3).

### Elo aplicado a futebol
- **Base**: Elo, A. (1978). *The Rating of Chessplayers*; adaptação a futebol:
  Hvattum, L. M. & Arntzen, H. (2010). *Using ELO ratings for match result prediction in association football*. Int. J. of Forecasting, 26(3).
- Rating contínuo, atualizado após cada jogo, sensível à **margem de gols**.

### Monte Carlo (simulação de temporada)
- Reamostragem de 10.000 temporadas a partir das probabilidades por jogo para
  estimar título/rebaixamento/vagas. Padrão em projeções (FiveThirtyEight SPI).

---

## 2. Expected Goals (xG) e Expected Threat (xT)

### xG
- Classificador de qualidade de finalização (XGBoost) com features de posição,
  ângulo, distância, parte do corpo e situação.
- **Base conceitual**: Rathke, A. (2017). *An examination of expected goals...*;
  metodologia consolidada por Opta/StatsBomb. Métrica de validação: AUC e log-loss,
  com **walk-forward** por temporada.

### xT — Expected Threat
- **Base**: Singh, K. (2019). *Introducing Expected Threat (xT)*. Valor posicional
  da posse via grade de transição de probabilidade de gol.

---

## 3. Apostas (betting)

### Detecção de valor (value betting)
- Compara a probabilidade do modelo com a **probabilidade implícita sem margem**
  da casa. **Base**: Dixon & Coles (1997, acima) sobre ineficiências de mercado;
  Constantinou, A. & Fenton, N. (2012). *Solving the problem of inadequate scoring rules...*.

### Critério de Kelly (fracionado)
- **Base**: Kelly, J. L. (1956). *A New Interpretation of Information Rate*. Bell System Tech. J.
- Usamos **quarter-Kelly** (¼) — reduz drawdown e o risco de ruína por erro de
  estimativa (Thorp, E. *The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market*, 2006).

### Validação e calibração
- **Walk-forward / purga**: López de Prado, M. (2018). *Advances in Financial Machine Learning*, cap. 7 (purging + embargo para evitar vazamento).
- **Qualidade da probabilidade**: Brier, G. W. (1950). *Verification of forecasts expressed in terms of probability*; diagramas de confiabilidade (reliability) para calibração.

---

## 4. Scouting, tática e outros

- **Similaridade de jogadores**: K-Means + similaridade de cosseno sobre métricas
  por 90'. Inspiração em descoberta de "papéis" (player roles) — cf. Decroos & Davis (2020), *Player Vectors*.
- **Redes de passes**: teoria de redes aplicada ao futebol — Peña, J. L. & Touchette, H. (2012). *A network theory analysis of football strategies*.
- **Pressing/PPDA**: definição de intensidade defensiva (passes permitidos por ação defensiva) — métrica Opta consolidada.
- **Árbitros**: perfis estatísticos de cartões/faltas por árbitro (base descritiva).
- **Risco de lesão**: carga aguda:crônica (ACWR) — Gabbett, T. (2016). *The training-injury prevention paradox*. BJSM.

---

## 5. Aprendizado contínuo (paper trading)

O sistema de teste com dinheiro simulado aprende com acertos e erros:
- **Calibração online por faixa** (isotonic-lite com encolhimento bayesiano):
  ajusta a probabilidade do modelo na direção da taxa de acerto observada.
- **Adaptação de limiares** (edge/confiança) conforme o desempenho realizado,
  para manter/melhorar o nível de acerto ao longo do tempo.
- Métricas honestas: ROI, yield, win rate, **expectancy**, Brier e drawdown.

---

## Dados (atualização)

- Incluído no repositório: **amostra do Brasileirão 2024** (jogos, classificação,
  stats de jogadores) — suficiente para treinar/validar e testar o pipeline.
- Para **temporada corrente** e outras ligas, conecte uma fonte ao vivo
  (API-Football, football-data.org) preenchendo as chaves no `.env`. Os modelos
  não mudam — só passam a treinar/prever sobre dados atuais.

> Importante: nenhum dado é inventado. Ampliar a base = plugar uma fonte real,
> não fabricar partidas.
