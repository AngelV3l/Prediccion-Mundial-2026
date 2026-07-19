"""Fit the Dixon-Coles model on the historical matches and save its parameters.

Usage:
    python scripts/03_fit_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldcup import config
from worldcup.model import fit_dixon_coles

# The tournament we are predicting; used as the reference date for time decay.
TARGET_YEAR = 2026


def main() -> None:
    matches = pd.read_csv(config.CLEAN_HISTORICAL)
    model = fit_dixon_coles(matches, reference_year=TARGET_YEAR)
    model.to_json(config.MODEL_PARAMS)
    print(f"Fitted {len(model.teams)} teams -> {config.MODEL_PARAMS}")
    print(f"home advantage = {model.home_advantage:.3f}   rho = {model.rho:.3f}\n")

    # A high defence rating is good: it is subtracted from the opponent's goal
    # rate, so overall strength is attack + defence.
    ranking = sorted(
        model.teams, key=lambda t: model.attack[t] + model.defence[t], reverse=True
    )
    print("Top 10 sides by net rating (attack + defence):")
    for team in ranking[:10]:
        net = model.attack[team] + model.defence[team]
        print(
            f"  {team:20s} att {model.attack[team]:+.2f}   "
            f"def {model.defence[team]:+.2f}   net {net:+.2f}"
        )


if __name__ == "__main__":
    main()
