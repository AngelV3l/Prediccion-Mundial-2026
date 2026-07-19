"""Simulate the 2026 World Cup, save probabilities, back-test and draw figures.

Usage:
    python scripts/04_simulate_tournament.py [--sims 10000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldcup import config, simulation
from worldcup.model import DixonColesModel

STAGE_COLUMNS = [
    ("round_of_32", simulation.R32),
    ("round_of_16", simulation.R16),
    ("quarter_final", simulation.QUARTER),
    ("semi_final", simulation.SEMI),
    ("final", simulation.FINAL),
]

# Palette (validated default from the dataviz skill).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def back_test(probs: pd.DataFrame, knockout: pd.DataFrame) -> pd.DataFrame:
    """Compare predicted stage probabilities with what actually happened in 2026."""
    actual = simulation.actual_reached_depths(knockout)
    records = []
    for _, row in probs.iterrows():
        depth = actual.get(row["team"], simulation.GROUP)
        for column, stage in STAGE_COLUMNS:
            records.append(
                {
                    "team": row["team"],
                    "stage": column,
                    "predicted": row[column],
                    "actual": int(depth >= stage),
                }
            )
    return pd.DataFrame(records)


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=0)


def figure_title_odds(probs: pd.DataFrame, path: Path, top: int = 15) -> None:
    import matplotlib.pyplot as plt

    top_teams = probs.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    values = top_teams["champion"] * 100
    ax.barh(top_teams["team"], values, color=BLUE, height=0.68, zorder=3)
    for y, v in enumerate(values):
        ax.text(v + 0.15, y, f"{v:.1f}%", va="center", ha="left", color=INK, fontsize=9)
    _style_axes(ax)
    ax.set_xlim(0, values.max() * 1.15)
    ax.set_xlabel("Probability of winning the tournament (%)", color=MUTED, fontsize=10)
    ax.set_title(
        "2026 World Cup title odds", color=INK, fontsize=15, fontweight="bold", loc="left"
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def figure_strengths(model: DixonColesModel, groups: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    teams = list(groups["team"])
    att = np.array([model.attack.get(t, 0.0) for t in teams])
    dfn = np.array([model.defence.get(t, 0.0) for t in teams])
    net = att + dfn

    fig, ax = plt.subplots(figsize=(8, 7), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.scatter(att, dfn, s=42, color=BLUE, alpha=0.85, zorder=3, edgecolor=SURFACE, linewidth=0.6)
    # Label the strongest sides to avoid clutter.
    for t, a, d, n in sorted(zip(teams, att, dfn, net), key=lambda z: z[3], reverse=True)[:16]:
        ax.annotate(t, (a, d), fontsize=8, color=INK, xytext=(4, 3), textcoords="offset points")
    _style_axes(ax)
    ax.set_xlabel("Attack rating  (higher scores more)", color=MUTED, fontsize=10)
    ax.set_ylabel("Defence rating  (higher concedes less)", color=MUTED, fontsize=10)
    ax.set_title(
        "Estimated team strength, 2026 participants",
        color=INK, fontsize=15, fontweight="bold", loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def figure_backtest(bt: pd.DataFrame, path: Path, n_bins: int = 5) -> float:
    """Reliability plot pooling all (team, stage) predictions; returns Brier score."""
    import matplotlib.pyplot as plt

    p = bt["predicted"].to_numpy()
    a = bt["actual"].to_numpy()
    brier = float(np.mean((p - a) ** 2))

    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.sum():
            xs.append(p[mask].mean())
            ys.append(a[mask].mean())

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 4)), label="Perfect calibration")
    ax.plot(xs, ys, marker="o", color=BLUE, linewidth=2, markersize=8,
            markeredgecolor=SURFACE, label="Model")
    _style_axes(ax)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted probability of reaching a stage", color=MUTED, fontsize=10)
    ax.set_ylabel("Observed frequency (2026)", color=MUTED, fontsize=10)
    ax.set_title(
        f"Back-test calibration  (Brier = {brier:.3f})",
        color=INK, fontsize=14, fontweight="bold", loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return brier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=10000, help="Number of simulated tournaments.")
    args = parser.parse_args()

    model = DixonColesModel.from_json(config.MODEL_PARAMS)
    groups = pd.read_csv(config.CLEAN_GROUPS_2026)
    knockout = pd.read_csv(config.CLEAN_FIXTURE_2026)

    print(f"Running {args.sims} simulated tournaments...")
    probs = simulation.monte_carlo(groups, model, n_sims=args.sims)
    probs.to_csv(config.SIM_RESULTS, index=False)
    print(f"Saved probabilities -> {config.SIM_RESULTS}\n")

    show = probs.head(12).copy()
    for col in ["round_of_16", "quarter_final", "semi_final", "final", "champion"]:
        show[col] = (show[col] * 100).round(1)
    print("Title contenders (%):")
    print(show[["team", "group", "round_of_16", "quarter_final", "semi_final",
                "final", "champion"]].to_string(index=False))

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_title_odds(probs, config.FIGURES_DIR / "title_odds.png")
    figure_strengths(model, groups, config.FIGURES_DIR / "team_strengths.png")
    brier = figure_backtest(back_test(probs, knockout), config.FIGURES_DIR / "backtest_calibration.png")
    print(f"\nBack-test Brier score (reaching each stage): {brier:.3f}")
    print(f"Figures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
