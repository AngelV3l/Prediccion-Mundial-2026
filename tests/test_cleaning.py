import math

import pandas as pd

from worldcup import cleaning


def test_parse_score_simple():
    assert cleaning.parse_score("3–1") == (3.0, 1.0)


def test_parse_score_extra_time_takes_first_pair():
    assert cleaning.parse_score("1–1 (a.e.t.)") == (1.0, 1.0)
    assert cleaning.parse_score("0–1 (a.e.t./g.g.)") == (0.0, 1.0)


def test_parse_score_walkover_and_placeholder_are_nan():
    for bad in ("w/o", "Match 104", ""):
        home, away = cleaning.parse_score(bad)
        assert math.isnan(home) and math.isnan(away)


def test_normalize_team_strips_host_and_footnotes():
    assert cleaning.normalize_team("Mexico (H)") == "Mexico"
    assert cleaning.normalize_team("Ghana[a]") == "Ghana"
    assert cleaning.normalize_team("\xa0Canada ") == "Canada"


def test_normalize_team_applies_aliases():
    assert cleaning.normalize_team("West Germany") == "Germany"
    assert cleaning.normalize_team("Zaire") == "DR Congo"


def test_clean_historical_drops_walkovers_and_types_goals():
    raw = pd.DataFrame(
        {
            "home": ["Brazil", "Sweden"],
            "score": ["2–1", "w/o"],
            "away": ["Italy", "Austria"],
            "year": [1970, 1938],
        }
    )
    out = cleaning.clean_historical(raw)
    assert len(out) == 1
    assert out.iloc[0]["HomeGoals"] == 2 and out.iloc[0]["AwayGoals"] == 1
    assert out["HomeGoals"].dtype.kind == "i"


def test_clean_2026_knockout_labels_stages():
    raw = pd.DataFrame(
        {"home": [f"H{i}" for i in range(32)],
         "away": [f"A{i}" for i in range(32)],
         "score": ["1–0"] * 32}
    )
    out = cleaning.clean_2026_knockout(raw)
    assert (out["stage"] == "Round of 32").sum() == 16
    assert (out["stage"] == "Final").sum() == 1
