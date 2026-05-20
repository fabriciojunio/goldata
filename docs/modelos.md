# Modelos de Machine Learning — GolData

## 1. Expected Goals (xG)

**Algoritmo**: XGBoost Classifier  
**Features**: posição (x, y), ângulo, distância ao gol, parte do corpo, situação (chute livre, pênalti, etc.)  
**Métricas**: AUC-ROC ≥ 0.80, log-loss < 0.45  
**Validação**: Walk-Forward com temporadas históricas

## 2. Predição de Resultados

### Modelo Poisson
Modela gols marcados por cada time como distribuições Poisson independentes com força de ataque/defesa.

### Dixon-Coles
Extensão do Poisson com ajuste para scorelines baixos (0-0, 1-0) que são subestimados.

### Modelo Elo
Rating contínuo que se atualiza após cada partida, considerando margem de gols.

### Monte Carlo
Simula 10.000 temporadas para estimar probabilidades de título, rebaixamento e vaga em copas.

## 3. Scouting — Similaridade de Jogadores

**Algoritmo**: K-Means + Similaridade de Cosseno  
**Features**: métricas por 90 minutos (gols, assistências, xG, dribles, passes chave, etc.)  
**Uso**: encontrar jogadores com perfil similar para transferências
