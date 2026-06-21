"""
Mercados derivados do modelo de gols (Dixon-Coles / Poisson).

A partir dos gols esperados (lambda) de cada time, reconstrói a matriz de
placares e calcula a probabilidade dos principais mercados de apostas:

  • 1X2 (Match Winner)
  • Over/Under de gols (qualquer linha: 1.5, 2.5, 3.5...)
  • BTTS (Ambas Marcam / Both Teams To Score)
  • Handicap Asiático (linhas inteiras, meia e quartos)

Modelos de gols costumam ter MAIS edge em Over/Under e Handicap do que no 1X2,
onde o mercado é mais eficiente. Por isso estes mercados foram adicionados.

REFERÊNCIAS: Dixon & Coles (1997); Karlis & Ntzoufras (2003) para o Poisson
bivariado; Maher (1982) para o modelo base de gols.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson


def _tau(h: int, a: int, lh: float, la: float, rho: float) -> float:
    """Correção Dixon-Coles para placares baixos (dependência 0-0,1-0,0-1,1-1)."""
    if h == 0 and a == 0:
        return 1.0 - lh * la * rho
    if h == 1 and a == 0:
        return 1.0 + la * rho
    if h == 0 and a == 1:
        return 1.0 + lh * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(home_xg: float, away_xg: float, rho: float = 0.0, max_goals: int = 12) -> np.ndarray:
    """Matriz P[h, a] de probabilidade de cada placar (com correção DC opcional)."""
    lh = max(1e-6, float(home_xg))
    la = max(1e-6, float(away_xg))
    h = poisson.pmf(np.arange(max_goals + 1), lh)
    a = poisson.pmf(np.arange(max_goals + 1), la)
    m = np.outer(h, a)
    if rho:
        for i in (0, 1):
            for j in (0, 1):
                m[i, j] *= max(0.0, _tau(i, j, lh, la, rho))
    s = m.sum()
    return m / s if s > 0 else m


def prob_1x2(m: np.ndarray) -> dict[str, float]:
    home = float(np.tril(m, -1).sum())   # h > a
    draw = float(np.trace(m))            # h == a
    away = float(np.triu(m, 1).sum())    # h < a
    return {"home_win": round(home, 4), "draw": round(draw, 4), "away_win": round(away, 4)}


def prob_over_under(m: np.ndarray, line: float = 2.5) -> dict[str, float]:
    """P(total de gols > line) e P(< line). Linha .5 não tem push."""
    n = m.shape[0]
    over = 0.0
    for h in range(n):
        for a in range(n):
            if h + a > line:
                over += m[h, a]
    over = float(over)
    return {"over": round(over, 4), "under": round(1.0 - over, 4)}


def prob_btts(m: np.ndarray) -> dict[str, float]:
    """Ambas marcam: P(h>=1 e a>=1)."""
    yes = float(m[1:, 1:].sum())
    return {"yes": round(yes, 4), "no": round(1.0 - yes, 4)}


def prob_asian_handicap(m: np.ndarray, line: float) -> dict[str, float]:
    """
    Handicap asiático para o MANDANTE na 'line' (ex.: -0.5, +1.0, -0.25).

    Retorna P(home cobre), P(away cobre) e P(push) já líquidas para linhas de
    quarto (split). Para linhas .0 há possibilidade de push (empate no handicap).
    """
    n = m.shape[0]
    # margem de gols do mandante (h - a)
    diff_probs: dict[int, float] = {}
    for h in range(n):
        for a in range(n):
            d = h - a
            diff_probs[d] = diff_probs.get(d, 0.0) + float(m[h, a])

    def settle(linha: float) -> tuple[float, float, float]:
        win = lose = push = 0.0
        for d, p in diff_probs.items():
            res = d + linha
            if res > 0:
                win += p
            elif res < 0:
                lose += p
            else:
                push += p
        return win, lose, push

    # Linha de quarto (ex.: -0.25, -0.75): metade em cada linha vizinha.
    if abs((line * 2) % 1) > 1e-9:  # é quarto (x.25 / x.75)
        baixo = line - 0.25
        alto = line + 0.25
        w1, l1, p1 = settle(baixo)
        w2, l2, p2 = settle(alto)
        # push numa metade devolve aquela metade (meio-ganho/meia-perda).
        home = (w1 + w2) / 2 + (p1 + p2) / 4
        away = (l1 + l2) / 2 + (p1 + p2) / 4
        return {"home": round(home, 4), "away": round(away, 4), "push": 0.0}

    win, lose, push = settle(line)
    return {"home": round(win, 4), "away": round(lose, 4), "push": round(push, 4)}


@dataclass
class MarketProbs:
    """Probabilidades de todos os mercados para um jogo."""
    one_x_two: dict[str, float]
    over_under_25: dict[str, float]
    btts: dict[str, float]


def all_markets(home_xg: float, away_xg: float, rho: float = 0.0) -> MarketProbs:
    """Atalho: calcula 1X2 + O/U 2.5 + BTTS de uma vez."""
    m = score_matrix(home_xg, away_xg, rho)
    return MarketProbs(
        one_x_two=prob_1x2(m),
        over_under_25=prob_over_under(m, 2.5),
        btts=prob_btts(m),
    )
