"""Project-wide paths and tournament constants.

Centralising these here keeps the scripts, notebook and tests in sync and avoids
hard-coded paths scattered across the codebase.
"""
from __future__ import annotations

from pathlib import Path

# --- Filesystem layout -------------------------------------------------------
# config.py lives at src/worldcup/config.py, so the project root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw scraped snapshots
RAW_HISTORICAL = RAW_DIR / "worldcup_historical_raw.csv"
RAW_GROUPS_2026 = RAW_DIR / "worldcup_2026_groups_raw.csv"
RAW_FIXTURE_2026 = RAW_DIR / "worldcup_2026_fixture_raw.csv"

# Processed datasets
CLEAN_HISTORICAL = PROCESSED_DIR / "matches_historical.csv"
CLEAN_GROUPS_2026 = PROCESSED_DIR / "groups_2026.csv"
CLEAN_FIXTURE_2026 = PROCESSED_DIR / "fixture_2026.csv"
MODEL_PARAMS = PROCESSED_DIR / "dixon_coles_params.json"
SIM_RESULTS = PROCESSED_DIR / "simulation_probabilities.csv"

# --- Historical data ---------------------------------------------------------
# All World Cup editions available on Wikipedia with match-level results.
WORLD_CUP_YEARS = [
    1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978,
    1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022,
]

# --- 2026 tournament format --------------------------------------------------
# 48 teams, 12 groups of 4. The top two of each group plus the eight best
# third-placed teams advance to a round of 32.
N_GROUPS_2026 = 12
GROUP_LETTERS_2026 = list("ABCDEFGHIJKL")
TEAMS_PER_GROUP = 4
N_BEST_THIRDS = 8

# Points awarded in the group stage.
WIN_POINTS = 3
DRAW_POINTS = 1
LOSS_POINTS = 0

# --- 2026 knockout bracket template ------------------------------------------
# Reverse-engineered from the official bracket: each actual Round-of-32 team was
# mapped back to its (group, group-position) using the published group standings.
# This is more reliable than transcribing the slot labels by hand.
#
# Each Round-of-32 match is a pair of "slots". A slot is one of:
#   ("W", "A")  -> winner of group A
#   ("R", "A")  -> runner-up of group A
#   ("T", "A")  -> a best-third-placed team; "A" is the reference group of the
#                  slot in the real 2026 bracket, used only to keep the
#                  third-placed allocation stable and reproducible (see
#                  simulation.assign_thirds_to_slots).
R32_TEMPLATE = [
    (("R", "A"), ("R", "B")),   # M73
    (("W", "C"), ("R", "F")),   # M74
    (("W", "E"), ("T", "D")),   # M75
    (("W", "F"), ("R", "C")),   # M76
    (("R", "E"), ("R", "I")),   # M77
    (("W", "I"), ("T", "F")),   # M78
    (("W", "A"), ("T", "E")),   # M79
    (("W", "L"), ("T", "K")),   # M80
    (("W", "G"), ("T", "I")),   # M81
    (("W", "D"), ("T", "B")),   # M82
    (("W", "H"), ("R", "J")),   # M83
    (("R", "K"), ("R", "L")),   # M84
    (("W", "B"), ("T", "J")),   # M85
    (("R", "D"), ("R", "G")),   # M86
    (("W", "J"), ("R", "H")),   # M87
    (("W", "K"), ("T", "L")),   # M88
]

# Feeder tree, as index pairs into the previous round's match list. Derived by
# tracing the winners of the real 2026 tournament, so it reproduces the exact
# (non-consecutive) bracket layout that keeps group rivals apart.
R16_PAIRS = [(0, 3), (2, 5), (1, 4), (6, 7), (10, 11), (8, 9), (13, 14), (12, 15)]
QF_PAIRS = [(0, 1), (4, 5), (2, 3), (6, 7)]
SF_PAIRS = [(0, 1), (2, 3)]
FINAL_PAIR = (0, 1)
