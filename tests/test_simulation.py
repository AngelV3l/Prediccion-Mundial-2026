import numpy as np
import pandas as pd
import pytest

from worldcup import config, simulation
from worldcup.model import DixonColesModel


def _toy_groups():
    rows = []
    for letter in config.GROUP_LETTERS_2026:
        for pos in range(1, 5):
            rows.append({"group": letter, "position": pos, "team": f"{letter}{pos}"})
    return pd.DataFrame(rows)


def _flat_model(teams):
    # All teams equal -> a clean structural test of the bracket plumbing.
    return DixonColesModel(
        attack={t: 0.0 for t in teams},
        defence={t: 0.0 for t in teams},
        home_advantage=0.0,
        rho=0.0,
        teams=teams,
    )


def test_group_fixtures_are_round_robins():
    fixtures = simulation.build_group_fixtures(_toy_groups())
    assert len(fixtures) == config.N_GROUPS_2026
    assert all(len(pairs) == 6 for pairs in fixtures.values())


def test_third_slots_cover_every_group_once():
    slots = simulation._third_slots()
    assert len(slots) == config.N_BEST_THIRDS
    refs = [ref for _, _, ref in slots]
    assert refs == sorted(refs)  # returned sorted by reference group
    assert len(set(refs)) == config.N_BEST_THIRDS


def test_monte_carlo_probabilities_are_consistent():
    groups = _toy_groups()
    model = _flat_model(list(groups["team"]))
    probs = simulation.monte_carlo(groups, model, n_sims=300, seed=0)

    assert len(probs) == 48
    # Exactly 32 teams reach the round of 32 every tournament.
    assert probs["round_of_32"].sum() == pytest.approx(32.0)
    # Exactly one champion per tournament.
    assert probs["champion"].sum() == pytest.approx(1.0)
    # Monotonic funnel: reaching a later stage is never more likely than an earlier one.
    for _, row in probs.iterrows():
        assert row["round_of_32"] >= row["round_of_16"] >= row["quarter_final"]
        assert row["quarter_final"] >= row["semi_final"] >= row["final"] >= row["champion"]


def test_actual_reached_depths():
    knockout = pd.DataFrame(
        {
            "stage": ["Round of 32", "Final", "Final"],
            "home": ["Minnow", "Spain", "Spain"],
            "away": ["Giant", "Argentina", "Argentina"],
            "home_goals": [0, np.nan, np.nan],
            "away_goals": [1, np.nan, np.nan],
        }
    )
    reached = simulation.actual_reached_depths(knockout)
    assert reached["Spain"] == simulation.FINAL
    assert reached["Minnow"] == simulation.R32
