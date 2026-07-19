import numpy as np
import pandas as pd
import pytest

from worldcup.model import DixonColesModel, fit_dixon_coles


@pytest.fixture
def toy_matches():
    # "Strong" beats "Weak" repeatedly; "Mid" sits in between.
    rows = []
    for year in range(2000, 2020, 4):
        rows += [
            {"HomeTeam": "Strong", "AwayTeam": "Weak", "Year": year, "HomeGoals": 3, "AwayGoals": 0},
            {"HomeTeam": "Strong", "AwayTeam": "Mid", "Year": year, "HomeGoals": 2, "AwayGoals": 1},
            {"HomeTeam": "Mid", "AwayTeam": "Weak", "Year": year, "HomeGoals": 2, "AwayGoals": 0},
        ]
    return pd.DataFrame(rows)


def test_attack_ratings_sum_to_zero(toy_matches):
    model = fit_dixon_coles(toy_matches, reference_year=2020, ridge=0.5)
    assert abs(sum(model.attack.values())) < 1e-6


def test_fit_recovers_strength_ordering(toy_matches):
    model = fit_dixon_coles(toy_matches, reference_year=2020, ridge=0.5)
    net = {t: model.attack[t] + model.defence[t] for t in model.teams}
    assert net["Strong"] > net["Mid"] > net["Weak"]


def test_scoreline_matrix_is_a_distribution():
    model = DixonColesModel(
        attack={"A": 0.3, "B": -0.3},
        defence={"A": 0.1, "B": -0.1},
        home_advantage=0.2,
        rho=-0.1,
        teams=["A", "B"],
    )
    matrix = model.scoreline_matrix("A", "B", max_goals=15)
    assert matrix.shape == (16, 16)
    assert matrix.min() >= 0
    assert model.match_probabilities("A", "B") == pytest.approx(
        (
            np.tril(matrix, -1).sum(),
            np.trace(matrix),
            np.triu(matrix, 1).sum(),
        )
    )
    assert sum(model.match_probabilities("A", "B")) == pytest.approx(1.0)


def test_unknown_team_gets_average_ratings():
    model = DixonColesModel(attack={"A": 0.5}, defence={"A": 0.5}, teams=["A"])
    # A debutant with no history is average, not unbeatable-or-hopeless.
    lam, mu = model.expected_goals("Debutant", "Other", neutral=True)
    assert lam == pytest.approx(1.0)
    assert mu == pytest.approx(1.0)


def test_json_round_trip(tmp_path, toy_matches):
    model = fit_dixon_coles(toy_matches, reference_year=2020, ridge=0.5)
    path = tmp_path / "params.json"
    model.to_json(path)
    reloaded = DixonColesModel.from_json(path)
    assert reloaded.attack == model.attack
    assert reloaded.rho == model.rho
