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

from worldcup import config, evaluation, simulation
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
    """Compare predicted stage probabilities with what actually happened in 2026.

    Adds the base-rate ``reference`` forecast (share of the field reaching each
    stage) so the model's Brier score has something to be compared with.
    """
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
    bt = pd.DataFrame(records)
    bt["reference"] = evaluation.base_rate_probabilities(bt["stage"], n_teams=len(probs))
    return bt


def back_test_summary(bt: pd.DataFrame) -> pd.DataFrame:
    """Brier score of the model and the reference, and the skill score, per stage and overall."""
    rows = []
    groups = [(stage, bt[bt["stage"] == stage]) for stage, _ in STAGE_COLUMNS] + [("all", bt)]
    for stage, d in groups:
        rows.append(
            {
                "stage": stage,
                "n": len(d),
                "brier_model": evaluation.brier_score(d["predicted"], d["actual"]),
                "brier_reference": evaluation.brier_score(d["reference"], d["actual"]),
                "skill_score": evaluation.brier_skill_score(
                    d["predicted"], d["actual"], d["reference"]
                ),
            }
        )
    return pd.DataFrame(rows)


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


def figure_backtest(
    bt: pd.DataFrame, summary: pd.DataFrame, ci: tuple[float, float], path: Path, n_bins: int = 5
) -> None:
    """Reliability plot pooling all (team, stage) predictions, with bin sizes and skill score."""
    import matplotlib.pyplot as plt

    table = evaluation.calibration_table(bt["predicted"], bt["actual"], n_bins=n_bins)
    overall = summary.set_index("stage").loc["all"]

    fig, ax = plt.subplots(figsize=(6.5, 6.9), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 4)), label="Perfect calibration")
    ax.plot(table["mean_predicted"], table["observed"], marker="o", color=BLUE, linewidth=2,
            markersize=8, markeredgecolor=SURFACE, label="Model")
    for x, y, n in zip(table["mean_predicted"], table["observed"], table["n"]):
        ax.annotate(f"n={n}", (x, y), xytext=(8, -12), textcoords="offset points",
                    fontsize=9, color=INK)
    _style_axes(ax)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Predicted probability of reaching a stage", color=MUTED, fontsize=10)
    ax.set_ylabel("Observed frequency (2026)", color=MUTED, fontsize=10)
    fig.suptitle("Back-test calibration", x=0.08, ha="left", color=INK, fontsize=14,
                 fontweight="bold")
    ax.set_title(
        f"Brier {overall['brier_model']:.3f} vs {overall['brier_reference']:.3f} for the "
        f"base-rate reference\nSkill score {overall['skill_score']:+.2f} "
        f"(95% team bootstrap {ci[0]:.2f} to {ci[1]:.2f}), n = {int(overall['n'])} predictions",
        color=MUTED, fontsize=9.5, loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


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
    bt = back_test(probs, knockout)
    summary = back_test_summary(bt)
    ci = evaluation.bootstrap_skill_score(bt)
    figure_backtest(bt, summary, ci, config.FIGURES_DIR / "backtest_calibration.png")

    print("\nBack-test, model vs base-rate reference (Brier score, lower is better):")
    print(summary.round(3).to_string(index=False))
    print(f"Skill score 95% team-bootstrap interval: {ci[0]:.3f} to {ci[1]:.3f}")
    print("\nCalibration bins:")
    print(evaluation.calibration_table(bt["predicted"], bt["actual"]).round(3).to_string(index=False))
    print(f"\nFigures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
