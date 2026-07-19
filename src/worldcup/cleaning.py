"""Turn the raw scraped tables into tidy, model-ready datasets.

The main jobs are parsing scorelines robustly (extra-time, walkovers and the
odd unplayed match all appear in the raw data) and reconciling team names so a
2026 side is linked to its full World Cup history, including predecessor states.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Historical names mapped onto the modern side that inherits their record.
# These are deliberate, debatable domain choices (documented in the README):
# FIFA treats the Czech Republic as the successor to Czechoslovakia's record,
# and DR Congo to Zaire's; West Germany's record is folded into Germany.
TEAM_ALIASES = {
    "West Germany": "Germany",
    "Soviet Union": "Russia",
    "Czechoslovakia": "Czech Republic",
    "Yugoslavia": "Serbia",
    "FR Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Zaire": "DR Congo",
}

_HOST_MARKER = re.compile(r"\s*\(H\)\s*$")
_FOOTNOTE = re.compile(r"\[.*?\]")
_SCORE = re.compile(r"(\d+)\s*[–\-]\s*(\d+)")

# Knockout rounds in the order the matches appear in the raw bracket frame.
KNOCKOUT_STAGES = (
    ("Round of 32", 16),
    ("Round of 16", 8),
    ("Quarter-final", 4),
    ("Semi-final", 2),
    ("Third place", 1),
    ("Final", 1),
)


def normalize_team(name: str) -> str:
    """Canonicalise a team name: strip host/footnote markup, unify Unicode, alias."""
    text = unicodedata.normalize("NFC", str(name)).replace("\xa0", " ")
    text = _FOOTNOTE.sub("", text)
    text = _HOST_MARKER.sub("", text)
    text = text.strip()
    return TEAM_ALIASES.get(text, text)


def parse_score(score: str) -> tuple[float, float]:
    """Return the first ``goals-goals`` pair in a scoreline, or NaN if there is none.

    Handles ``"3–1"``, extra-time results like ``"1–1 (a.e.t.)"`` and gracefully
    skips walkovers (``"w/o"``) and placeholders for unplayed matches.
    """
    match = _SCORE.search(str(score))
    if not match:
        return (float("nan"), float("nan"))
    return (float(match.group(1)), float(match.group(2)))


def clean_historical(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the historical match snapshot into tidy, typed rows.

    Output columns: ``HomeTeam``, ``AwayTeam``, ``Year``, ``HomeGoals``,
    ``AwayGoals``. Matches without a numeric result (walkovers) are dropped.
    """
    df = raw.copy()
    goals = df["score"].apply(parse_score)
    df["HomeGoals"] = [g[0] for g in goals]
    df["AwayGoals"] = [g[1] for g in goals]
    df = df.dropna(subset=["HomeGoals", "AwayGoals"]).copy()

    df["HomeTeam"] = df["home"].apply(normalize_team)
    df["AwayTeam"] = df["away"].apply(normalize_team)
    df = df.rename(columns={"year": "Year"})
    df = df.astype({"HomeGoals": int, "AwayGoals": int, "Year": int})
    return df[["HomeTeam", "AwayTeam", "Year", "HomeGoals", "AwayGoals"]].reset_index(drop=True)


def clean_2026_groups(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the 2026 group standings into ``group``, ``position``, ``team``."""
    df = raw.copy()
    df["team"] = df["team"].apply(normalize_team)
    df["position"] = df["position"].astype(int)
    return df[["group", "position", "team"]].sort_values(
        ["group", "position"]
    ).reset_index(drop=True)


def clean_2026_knockout(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the actual 2026 knockout results and label each match with its stage.

    Output columns: ``stage``, ``home``, ``away``, ``home_goals``, ``away_goals``.
    Goals are NaN for any match not yet played (e.g. the final on the day it is
    scraped); the team names are still kept so the back-test can see who reached
    each round.
    """
    df = raw.copy()
    stages = []
    for name, size in KNOCKOUT_STAGES:
        stages.extend([name] * size)
    df = df.iloc[: len(stages)].copy()
    df["stage"] = stages[: len(df)]

    df["home"] = df["home"].apply(normalize_team)
    df["away"] = df["away"].apply(normalize_team)
    goals = df["score"].apply(parse_score)
    df["home_goals"] = [g[0] for g in goals]
    df["away_goals"] = [g[1] for g in goals]
    return df[["stage", "home", "away", "home_goals", "away_goals"]].reset_index(drop=True)
