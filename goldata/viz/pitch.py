"""Visualizações de campo de futebol com matplotlib."""

from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIELD_COLOR = "#1a7a3c"
LINE_COLOR = "white"
FIELD_LENGTH = 120.0
FIELD_WIDTH = 80.0


def draw_pitch(
    ax: plt.Axes | None = None,
    figsize: tuple = (12, 8),
    color: str = FIELD_COLOR,
    line_color: str = LINE_COLOR,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Desenha um campo de futebol no padrão StatsBomb (120x80).

    Args:
        ax: eixo matplotlib existente (cria novo se None)
        figsize: tamanho da figura
        color: cor do campo
        line_color: cor das linhas

    Returns:
        (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Campo
    ax.set_facecolor(color)
    ax.add_patch(patches.Rectangle(
        (0, 0), FIELD_LENGTH, FIELD_WIDTH,
        linewidth=2, edgecolor=line_color, facecolor=color,
    ))

    # Linha do meio
    ax.plot([60, 60], [0, 80], color=line_color, linewidth=1.5)

    # Círculo do meio
    circle_center = patches.Circle((60, 40), 9.15, color=line_color, fill=False, linewidth=1.5)
    ax.add_patch(circle_center)
    ax.plot(60, 40, "o", color=line_color, markersize=3)

    # Área grande - casa (direita)
    ax.add_patch(patches.Rectangle(
        (102, 18), 18, 44,
        linewidth=1.5, edgecolor=line_color, facecolor="none",
    ))
    # Área pequena - casa
    ax.add_patch(patches.Rectangle(
        (114, 30), 6, 20,
        linewidth=1.5, edgecolor=line_color, facecolor="none",
    ))
    # Gol - casa
    ax.add_patch(patches.Rectangle(
        (120, 36), 0, 8,
        linewidth=3, edgecolor=line_color, facecolor="none",
    ))
    # Pênalti - casa
    ax.plot(108, 40, "o", color=line_color, markersize=3)

    # Área grande - visitante (esquerda)
    ax.add_patch(patches.Rectangle(
        (0, 18), 18, 44,
        linewidth=1.5, edgecolor=line_color, facecolor="none",
    ))
    # Área pequena - visitante
    ax.add_patch(patches.Rectangle(
        (0, 30), 6, 20,
        linewidth=1.5, edgecolor=line_color, facecolor="none",
    ))
    # Gol - visitante
    ax.add_patch(patches.Rectangle(
        (0, 36), 0, 8,
        linewidth=3, edgecolor=line_color, facecolor="none",
    ))
    # Pênalti - visitante
    ax.plot(12, 40, "o", color=line_color, markersize=3)

    ax.set_xlim(-2, FIELD_LENGTH + 2)
    ax.set_ylim(-2, FIELD_WIDTH + 2)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig, ax


def plot_shot_map(
    shots_df: pd.DataFrame,
    title: str = "Shot Map",
    figsize: tuple = (12, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plota mapa de chutes com tamanho proporcional ao xG.

    Args:
        shots_df: DataFrame com x, y, xg, is_goal
        title: título do gráfico

    Returns:
        (fig, ax)
    """
    fig, ax = draw_pitch(figsize=figsize)

    if "x" not in shots_df.columns or "y" not in shots_df.columns:
        ax.set_title(title, color="white", fontsize=14)
        return fig, ax

    goals = shots_df[shots_df.get("is_goal", pd.Series([False] * len(shots_df))) == 1]
    non_goals = shots_df[shots_df.get("is_goal", pd.Series([False] * len(shots_df))) != 1]

    xg_col = shots_df.get("xg", pd.Series([0.1] * len(shots_df)))
    size_scale = 200

    if len(non_goals) > 0:
        xg_vals = non_goals.get("xg", pd.Series([0.1] * len(non_goals)))
        ax.scatter(
            non_goals["x"], non_goals["y"],
            s=xg_vals.fillna(0.1) * size_scale,
            c="white", alpha=0.6, edgecolors="white", linewidths=1,
            zorder=5, label="Chute",
        )

    if len(goals) > 0:
        xg_vals_g = goals.get("xg", pd.Series([0.1] * len(goals)))
        ax.scatter(
            goals["x"], goals["y"],
            s=xg_vals_g.fillna(0.1) * size_scale,
            c="yellow", alpha=0.9, edgecolors="white", linewidths=1.5,
            zorder=6, label="Gol", marker="*",
        )

    ax.set_title(title, color="white", fontsize=14, pad=10)
    return fig, ax


def plot_heatmap(
    events_df: pd.DataFrame,
    title: str = "Heatmap",
    figsize: tuple = (12, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plota heatmap de eventos no campo.

    Args:
        events_df: DataFrame com x, y

    Returns:
        (fig, ax)
    """
    fig, ax = draw_pitch(figsize=figsize)

    if "x" not in events_df.columns:
        ax.set_title(title, color="white", fontsize=14)
        return fig, ax

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("heat", ["#1a7a3c", "yellow", "red"])

    ax.hexbin(
        events_df["x"], events_df["y"],
        gridsize=20, cmap=cmap,
        alpha=0.6, mincnt=1, extent=[0, 120, 0, 80],
        zorder=4,
    )

    ax.set_title(title, color="white", fontsize=14, pad=10)
    return fig, ax
