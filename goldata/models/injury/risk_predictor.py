"""Modelo de risco de lesão baseado em carga de trabalho e histórico."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from goldata.exceptions import ModelNotTrainedError
from goldata.logging_config import get_logger

logger = get_logger(__name__)

INJURY_FEATURES = [
    "minutes_last_7_days",
    "minutes_last_30_days",
    "matches_last_30_days",
    "days_since_last_match",
    "age",
    "previous_injuries_12m",
    "high_intensity_actions_per_90",
    "sprints_per_90",
    "distance_covered_per_90",
    "minutes_this_season",
]

RISK_LABELS = {0: "Baixo", 1: "Médio", 2: "Alto", 3: "Crítico"}


@dataclass
class InjuryRiskReport:
    player_id: str
    risk_score: float        # 0-1
    risk_level: str          # Baixo / Médio / Alto / Crítico
    risk_factors: list[str]
    recommended_rest_days: int
    confidence: float


class InjuryRiskPredictor:
    """
    Prediz risco de lesão baseado em carga de trabalho e histórico do jogador.

    Fatores considerados:
    - Sobrecarga recente (minutos últimos 7/30 dias)
    - Acúmulo de jogos
    - Idade
    - Histórico de lesões anteriores
    - Intensidade de corridas/sprints
    """

    model_name = "InjuryRiskPredictor"

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=random_state
        )
        self.is_trained: bool = False
        self._feature_names = INJURY_FEATURES

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for f in INJURY_FEATURES:
            if f not in result.columns:
                result[f] = 0.0
        return result[INJURY_FEATURES].fillna(0.0)

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Treina o modelo. y deve ser inteiro 0-3 (nível de risco).
        """
        X_prep = self._prepare(X)
        X_scaled = self._scaler.fit_transform(X_prep)
        self._model.fit(X_scaled, y)
        self.is_trained = True
        accuracy = float(self._model.score(X_scaled, y))
        logger.info("injury_model_trained", accuracy=accuracy, n=len(X))
        return {"accuracy": round(accuracy, 4), "n_samples": len(X)}

    def predict_risk(self, player_data: dict) -> InjuryRiskReport:
        """
        Prediz risco de lesão para um jogador.

        Returns:
            InjuryRiskReport com score, nível e recomendações
        """
        if not self.is_trained:
            raise ModelNotTrainedError(self.model_name)

        df = pd.DataFrame([player_data])
        X_prep = self._prepare(df)
        X_scaled = self._scaler.transform(X_prep)

        proba = self._model.predict_proba(X_scaled)[0]
        risk_class = int(np.argmax(proba))
        risk_score = float(1 - proba[0])  # probabilidade de NÃO ser baixo risco invertida

        risk_factors = []
        mins_7 = float(player_data.get("minutes_last_7_days", 0))
        mins_30 = float(player_data.get("minutes_last_30_days", 0))
        age = float(player_data.get("age", 25))
        prev_injuries = int(player_data.get("previous_injuries_12m", 0))
        days_rest = float(player_data.get("days_since_last_match", 3))

        if mins_7 > 180:
            risk_factors.append("Sobrecarga semanal (>180 min em 7 dias)")
        if mins_30 > 800:
            risk_factors.append("Alto volume mensal (>800 min em 30 dias)")
        if age > 32:
            risk_factors.append("Idade avançada (>32 anos)")
        if prev_injuries >= 2:
            risk_factors.append(f"{prev_injuries} lesões nos últimos 12 meses")
        if days_rest < 2:
            risk_factors.append("Recuperação insuficiente (<2 dias desde último jogo)")

        rest_map = {0: 0, 1: 1, 2: 3, 3: 7}
        recommended_rest = rest_map.get(risk_class, 0)

        return InjuryRiskReport(
            player_id=str(player_data.get("player_id", "unknown")),
            risk_score=round(risk_score, 4),
            risk_level=RISK_LABELS.get(risk_class, "Desconhecido"),
            risk_factors=risk_factors,
            recommended_rest_days=recommended_rest,
            confidence=round(float(max(proba)), 4),
        )

    def predict_batch(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Prediz risco para múltiplos jogadores."""
        if not self.is_trained:
            raise ModelNotTrainedError(self.model_name)
        X_prep = self._prepare(players_df)
        X_scaled = self._scaler.transform(X_prep)
        classes = self._model.predict(X_scaled)
        probas = self._model.predict_proba(X_scaled)
        result = players_df.copy()
        result["risk_class"] = classes
        result["risk_level"] = [RISK_LABELS.get(c, "?") for c in classes]
        result["risk_score"] = (1 - probas[:, 0]).round(4)
        return result
