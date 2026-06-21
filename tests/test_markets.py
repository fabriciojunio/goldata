"""Testes dos mercados derivados (Over/Under, BTTS, Handicap Asiático)."""

import numpy as np
import pytest

from goldata.models.betting.markets import (
    all_markets,
    prob_1x2,
    prob_asian_handicap,
    prob_btts,
    prob_over_under,
    score_matrix,
)


def test_matriz_soma_um():
    m = score_matrix(1.5, 1.2)
    assert abs(m.sum() - 1.0) < 1e-6


def test_1x2_soma_um():
    m = score_matrix(1.6, 1.0)
    p = prob_1x2(m)
    assert abs(p["home_win"] + p["draw"] + p["away_win"] - 1.0) < 1e-3
    assert p["home_win"] > p["away_win"]  # mandante mais forte


def test_over_under_soma_um_e_monotonico():
    m = score_matrix(1.8, 1.5)
    ou25 = prob_over_under(m, 2.5)
    assert abs(ou25["over"] + ou25["under"] - 1.0) < 1e-6
    # Mais gols esperados => maior P(over)
    alto = prob_over_under(score_matrix(2.6, 2.2), 2.5)["over"]
    baixo = prob_over_under(score_matrix(0.8, 0.6), 2.5)["over"]
    assert alto > baixo


def test_linha_ou_maior_reduz_over():
    m = score_matrix(1.7, 1.3)
    assert prob_over_under(m, 1.5)["over"] > prob_over_under(m, 3.5)["over"]


def test_btts():
    m = score_matrix(1.5, 1.5)
    b = prob_btts(m)
    assert abs(b["yes"] + b["no"] - 1.0) < 1e-6
    # Times ofensivos => BTTS mais provável que dois times fracos
    forte = prob_btts(score_matrix(2.0, 2.0))["yes"]
    fraco = prob_btts(score_matrix(0.5, 0.5))["yes"]
    assert forte > fraco


def test_handicap_meia_linha_sem_push():
    m = score_matrix(1.8, 1.0)
    ah = prob_asian_handicap(m, -0.5)
    assert ah["push"] == 0.0
    assert abs(ah["home"] + ah["away"] - 1.0) < 1e-3


def test_handicap_linha_inteira_tem_push():
    m = score_matrix(1.5, 1.5)
    ah = prob_asian_handicap(m, 0.0)  # empate = push
    assert ah["push"] > 0.0


def test_handicap_quarto_distribui():
    m = score_matrix(1.6, 1.2)
    ah = prob_asian_handicap(m, -0.25)
    assert ah["push"] == 0.0
    assert 0.0 < ah["home"] < 1.0


def test_all_markets():
    mk = all_markets(1.7, 1.1, rho=0.05)
    assert set(mk.one_x_two) == {"home_win", "draw", "away_win"}
    assert set(mk.over_under_25) == {"over", "under"}
    assert set(mk.btts) == {"yes", "no"}
