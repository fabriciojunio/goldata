"""Validadores de dados do GolData."""

from goldata.exceptions import InvalidInputError


class DataValidator:
    """Valida dados de entrada antes de processar nos modelos."""

    @staticmethod
    def validate_shot_data(data: dict) -> bool:
        """Valida dados de um chute. Lança InvalidInputError se inválido."""
        x = data.get("x")
        y = data.get("y")
        xg = data.get("xg")

        if x is not None and not (0 <= x <= 120):
            raise InvalidInputError("x", x, "deve estar entre 0 e 120 (campo StatsBomb)")
        if y is not None and not (0 <= y <= 80):
            raise InvalidInputError("y", y, "deve estar entre 0 e 80 (campo StatsBomb)")
        if xg is not None and not (0 <= xg <= 1):
            raise InvalidInputError("xg", xg, "probabilidade deve estar entre 0 e 1")
        return True

    @staticmethod
    def validate_player_stats(data: dict) -> bool:
        """Valida métricas de jogador por 90 minutos."""
        per90_fields = ["goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90"]
        for field in per90_fields:
            value = data.get(field)
            if value is not None and value < 0:
                raise InvalidInputError(field, value, "métricas por 90min não podem ser negativas")
            if value is not None and value > 10:
                raise InvalidInputError(field, value, "valor irrealisticamente alto por 90min")
        return True

    @staticmethod
    def validate_match_result(home_score: int, away_score: int) -> bool:
        """Valida placar de partida."""
        if home_score < 0:
            raise InvalidInputError("home_score", home_score, "placar não pode ser negativo")
        if away_score < 0:
            raise InvalidInputError("away_score", away_score, "placar não pode ser negativo")
        if home_score > 20 or away_score > 20:
            raise InvalidInputError(
                "score", f"{home_score}x{away_score}", "placar irrealisticamente alto"
            )
        return True

    @staticmethod
    def validate_odds(odd: float) -> bool:
        """Valida odd de casa de apostas (deve ser >= 1.01)."""
        if odd < 1.0:
            raise InvalidInputError("odd", odd, "odd deve ser >= 1.0")
        if odd > 1000:
            raise InvalidInputError("odd", odd, "odd irrealisticamente alta")
        return True

    @staticmethod
    def validate_probability(prob: float, field: str = "probability") -> bool:
        """Valida que um valor é uma probabilidade válida [0, 1]."""
        if not (0 <= prob <= 1):
            raise InvalidInputError(field, prob, "probabilidade deve estar entre 0 e 1")
        return True
