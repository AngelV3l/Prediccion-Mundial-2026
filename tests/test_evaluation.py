import numpy as np
import pandas as pd
import pytest

from worldcup import evaluation


def test_brier_score_bounds():
    assert evaluation.brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert evaluation.brier_score([0.0, 1.0], [1, 0]) == 1.0
    assert evaluation.brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_base_rates_follow_the_48_team_format():
    stages = pd.Series(list(evaluation.STAGE_SLOTS))
    rates = evaluation.base_rate_probabilities(stages, n_teams=48)
    assert rates.tolist() == pytest.approx([32 / 48, 16 / 48, 8 / 48, 4 / 48, 2 / 48])


def test_skill_score_is_zero_for_the_reference_and_one_for_a_perfect_forecast():
    actual = np.array([1, 0, 1, 0, 0])
    reference = np.full(5, 0.4)
    assert evaluation.brier_skill_score(reference, actual, reference) == pytest.approx(0.0)
    assert evaluation.brier_skill_score(actual, actual, reference) == pytest.approx(1.0)


def test_calibration_table_counts_every_prediction_once():
    p = np.array([0.05, 0.1, 0.3, 0.55, 0.9, 1.0])
    a = np.array([0, 0, 1, 1, 1, 1])
    table = evaluation.calibration_table(p, a, n_bins=5)
    assert table["n"].sum() == len(p)
    top = table.iloc[-1]
    assert top["n"] == 2 and top["observed"] == 1.0  # p = 1.0 falls in the last bin


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(1)
    teams = np.repeat([f"T{i}" for i in range(40)], 5)
    predicted = rng.uniform(0, 1, teams.size)
    actual = (rng.uniform(0, 1, teams.size) < predicted).astype(int)
    bt = pd.DataFrame(
        {"team": teams, "predicted": predicted, "actual": actual, "reference": 0.5}
    )
    point = evaluation.brier_skill_score(bt.predicted, bt.actual, bt.reference)
    lo, hi = evaluation.bootstrap_skill_score(bt, n_boot=500)
    assert lo < point < hi
