"""Projeção de performance ao longo da carreira por posição e idade."""

import numpy as np
import pandas as pd

from goldata.logging_config import get_logger

logger = get_logger(__name__)

# Pico de performance por posição (anos de idade)
PEAK_WINDOWS: dict[str, tuple[int, int]] = {
    "FW": (25, 29),   # Atacantes
    "MF": (26, 30),   # Meias
    "DF": (27, 32),   # Defensores
    "GK": (28, 34),   # Goleiros
}

# Curva de desenvolvimento relativa (índice: 17-40 anos)
# Valores normalizados: 1.0 = pico
_AGE_START = 17
_AGE_CURVE_BASE = {
    "FW": [0.55, 0.65, 0.75, 0.85, 0.92, 0.97, 1.00, 1.00, 0.98, 0.95,
           0.90, 0.84, 0.76, 0.66, 0.55, 0.44, 0.33, 0.23, 0.15, 0.09, 0.05, 0.03, 0.01, 0.01],
    "MF": [0.50, 0.60, 0.70, 0.80, 0.88, 0.94, 0.98, 1.00, 1.00, 0.99,
           0.96, 0.91, 0.84, 0.75, 0.64, 0.53, 0.41, 0.30, 0.20, 0.12, 0.07, 0.04, 0.02, 0.01],
    "DF": [0.45, 0.55, 0.65, 0.75, 0.83, 0.90, 0.96, 0.99, 1.00, 1.00,
           0.99, 0.97, 0.93, 0.87, 0.79, 0.68, 0.56, 0.43, 0.31, 0.20, 0.12, 0.07, 0.03, 0.01],
    "GK": [0.40, 0.50, 0.60, 0.68, 0.76, 0.83, 0.89, 0.94, 0.97, 1.00,
           1.00, 1.00, 0.98, 0.95, 0.90, 0.83, 0.74, 0.62, 0.49, 0.36, 0.24, 0.14, 0.07, 0.03],
}


class PerformanceProjector:
    """
    Projeta a curva de performance de um jogador ao longo da carreira.

    Usa curvas de idade por posição baseadas em literatura de sports science.
    Referência: Dendir (2016), Kalén et al. (2019), TruMedia Networks.
    """

    def get_age_curve(self, position: str) -> np.ndarray:
        """
        Retorna curva de performance normalizada para uma posição.

        Args:
            position: "FW", "MF", "DF" ou "GK"

        Returns:
            Array com valores 0-1 para idades 17-40
        """
        pos = position.upper()
        if pos not in _AGE_CURVE_BASE:
            pos = "MF"  # fallback
        return np.array(_AGE_CURVE_BASE[pos])

    def get_peak_window(self, position: str) -> tuple[int, int]:
        """Retorna janela de pico (age_start, age_end) por posição."""
        pos = position.upper()
        return PEAK_WINDOWS.get(pos, (26, 30))

    def _get_multiplier(self, current_age: float, target_age: float, position: str) -> float:
        """Calcula multiplicador de performance entre duas idades."""
        curve = self.get_age_curve(position)
        def age_to_idx(age: float) -> int:
            return max(0, min(int(age) - _AGE_START, len(curve) - 1))

        current_val = curve[age_to_idx(current_age)]
        target_val = curve[age_to_idx(target_age)]
        if current_val == 0:
            return 1.0
        return float(target_val / current_val)

    def project_performance(
        self,
        player_stats: dict,
        target_age: int,
        position: str | None = None,
    ) -> dict:
        """
        Projeta as métricas de um jogador para uma idade futura.

        Args:
            player_stats: dict com métricas atuais e idade atual
            target_age: idade alvo para projeção
            position: posição do jogador (usa player_stats["position"] se None)

        Returns:
            dict com métricas projetadas
        """
        current_age = float(player_stats.get("age", 25))
        pos = position or str(player_stats.get("position", "MF"))
        multiplier = self._get_multiplier(current_age, target_age, pos)

        projected = {"age": target_age, "position": pos, "projection_multiplier": round(multiplier, 4)}

        performance_metrics = [
            "goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
            "shots_per_90", "key_passes_per_90", "dribbles_completed_per_90",
            "tackles_per_90", "pressures_per_90",
        ]
        for metric in performance_metrics:
            if metric in player_stats:
                projected[metric] = round(float(player_stats[metric]) * multiplier, 4)

        # Métricas de taxa (pass completion) não mudam tanto com idade
        stable_metrics = ["pass_completion_rate", "shot_accuracy"]
        for metric in stable_metrics:
            if metric in player_stats:
                projected[metric] = player_stats[metric]

        peak_start, peak_end = self.get_peak_window(pos)
        projected["in_peak_window"] = peak_start <= target_age <= peak_end
        projected["years_to_peak"] = max(0, peak_start - target_age)
        return projected

    def get_career_trajectory(
        self,
        player_stats: dict,
        position: str | None = None,
        age_range: tuple[int, int] = (17, 38),
    ) -> pd.DataFrame:
        """
        Retorna a trajetória de performance completa de um jogador.

        Returns:
            DataFrame com projeções para cada ano de carreira
        """
        current_age = float(player_stats.get("age", 25))
        pos = position or str(player_stats.get("position", "MF"))
        curve = self.get_age_curve(pos)
        peak_start, peak_end = self.get_peak_window(pos)

        rows = []
        for age in range(age_range[0], age_range[1] + 1):
            idx = max(0, min(age - _AGE_START, len(curve) - 1))
            rows.append({
                "age": age,
                "performance_index": round(float(curve[idx]), 4),
                "in_peak": peak_start <= age <= peak_end,
                "is_current_age": age == int(current_age),
            })
        return pd.DataFrame(rows)
