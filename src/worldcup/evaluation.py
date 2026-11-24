"""Back-test metrics: Brier score, a base-rate reference and calibration bins.

A Brier score on its own says little, so the model is compared with a
reference forecast that knows the tournament format but nothing about the
teams: every side gets the share of the field that reaches each stage
(32 of 48 reach the round of 32, 16 of 48 the round of 16, and so on).
The Brier skill score ``1 - BS_model / BS_reference`` is the fraction of the
reference's error that the model removes: 0 means no better than the format,
1 means perfect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Teams that reach each knockout stage in the 48-team format.
STAGE_SLOTS = {
    "round_of_32": 32,
    "round_of_16": 16,
    "quarter_final": 8,
    "semi_final": 4,
    "final": 2,
}


def brier_score(predicted, actual) -> float:
    """Mean squared difference between predicted probabilities and 0/1 outcomes."""
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actual, dtype=float)
    return float(np.mean((p - a) ** 2))


def base_rate_probabilities(stages: pd.Series, n_teams: int) -> pd.Series:
    """Reference forecast: the share of the field that reaches each stage."""
    return stages.map(STAGE_SLOTS) / n_teams


def brier_skill_score(predicted, actual, reference) -> float:
    """``1 - BS(predicted) / BS(reference)``; positive means better than the reference."""
    return 1.0 - brier_score(predicted, actual) / brier_score(reference, actual)


def calibration_table(predicted, actual, n_bins: int = 5) -> pd.DataFrame:
    """Bin predictions into equal-width bins and compare with observed frequencies.

    Returns one row per non-empty bin with the number of predictions ``n``, the
    mean predicted probability and the observed frequency.
    """
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actual, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            rows.append(
                {
                    "bin_low": lo,
                    "bin_high": hi,
                    "n": int(mask.sum()),
                    "mean_predicted": float(p[mask].mean()),
                    "observed": float(a[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_skill_score(
    bt: pd.DataFrame, n_boot: int = 2000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval for the Brier skill score, resampling whole teams.

    ``bt`` needs ``team``, ``predicted``, ``actual`` and ``reference`` columns.
    Teams are resampled rather than single predictions because a team's five
    stage outcomes are nested. Teams in one tournament are not fully
    independent either, so read the interval as approximate.
    """
    rng = np.random.default_rng(seed)
    by_team = [g[["predicted", "actual", "reference"]].to_numpy() for _, g in bt.groupby("team")]
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(by_team), len(by_team))
        sample = np.concatenate([by_team[j] for j in idx])
        scores[i] = brier_skill_score(sample[:, 0], sample[:, 1], sample[:, 2])
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(scores, [tail, 100 - tail])
    return float(lo), float(hi)
