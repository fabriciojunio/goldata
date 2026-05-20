"""Gráfico radar para comparação de perfis de jogadores."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch


def plot_player_radar(
    player_stats: dict,
    metrics: list[str] | None = None,
    title: str = "Player Radar",
    color: str = "#e74c3c",
    figsize: tuple = (8, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plota gráfico radar para um jogador.

    Args:
        player_stats: dict com métricas normalizadas [0, 1]
        metrics: lista de métricas a plotar (usa todas se None)
        title: título do gráfico
        color: cor do radar
        figsize: tamanho da figura

    Returns:
        (fig, ax)
    """
    if metrics is None:
        metrics = list(player_stats.keys())

    values = [float(player_stats.get(m, 0)) for m in metrics]
    values = [max(0.0, min(1.0, v)) for v in values]  # normalizar 0-1

    n = len(metrics)
    if n < 3:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Mínimo 3 métricas para radar", ha="center", va="center")
        return fig, ax

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # Grid
    for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot(
            angles_plot,
            [level] * (n + 1),
            color="gray", alpha=0.3, linewidth=0.8,
        )

    # Radar
    ax.fill(angles, values, color=color, alpha=0.25)
    ax.plot(angles_plot, values_plot, color=color, linewidth=2.5)
    ax.scatter(angles, values, color=color, s=60, zorder=5)

    # Labels
    ax.set_thetagrids(np.degrees(angles), metrics, fontsize=9, color="white")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color="gray")
    ax.tick_params(colors="white")
    ax.spines["polar"].set_color("gray")
    ax.grid(False)

    ax.set_title(title, color="white", fontsize=14, pad=20)
    return fig, ax


def plot_comparison_radar(
    player_a: dict,
    player_b: dict,
    metrics: list[str] | None = None,
    name_a: str = "Player A",
    name_b: str = "Player B",
    figsize: tuple = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Compara dois jogadores num radar duplo.
    """
    if metrics is None:
        metrics = list(player_a.keys())

    values_a = [max(0.0, min(1.0, float(player_a.get(m, 0)))) for m in metrics]
    values_b = [max(0.0, min(1.0, float(player_b.get(m, 0)))) for m in metrics]

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    for vals, color, name in [
        (values_a, "#e74c3c", name_a),
        (values_b, "#3498db", name_b),
    ]:
        vals_plot = vals + [vals[0]]
        angles_plot = angles + [angles[0]]
        ax.fill(angles, vals, color=color, alpha=0.15)
        ax.plot(angles_plot, vals_plot, color=color, linewidth=2, label=name)
        ax.scatter(angles, vals, color=color, s=50, zorder=5)

    ax.set_thetagrids(np.degrees(angles), metrics, fontsize=9, color="white")
    ax.set_ylim(0, 1)
    ax.tick_params(colors="white")
    ax.spines["polar"].set_color("gray")
    ax.grid(color="gray", alpha=0.2)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), labelcolor="white",
              facecolor="#1a1a2e", edgecolor="gray")
    ax.set_title(f"{name_a} vs {name_b}", color="white", fontsize=13, pad=20)
    return fig, ax
