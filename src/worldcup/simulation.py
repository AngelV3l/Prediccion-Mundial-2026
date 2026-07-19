"""Monte Carlo simulation of the 2026 World Cup from a fitted goal model.

Each simulated tournament plays the twelve four-team groups as round-robins,
ranks the group standings and the eight best third-placed teams, fills the
official round-of-32 bracket template (see :mod:`worldcup.config`) and plays the
knockout tree to a champion. Repeating this thousands of times turns the goal
model into probabilities: how likely each team is to escape its group, reach the
final, and win the trophy.

Group and knockout matches are played at a neutral venue. Knockout ties are
resolved by a strength-weighted coin flip standing in for a penalty shootout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .model import DixonColesModel

# Stage depth reached, used both to track progress and to aggregate probabilities.
GROUP = 0
R32 = 1
R16 = 2
QUARTER = 3
SEMI = 4
FINAL = 5
CHAMPION = 6

STAGE_LABELS = {
    R32: "Round of 32",
    R16: "Round of 16",
    QUARTER: "Quarter-final",
    SEMI: "Semi-final",
    FINAL: "Final",
    CHAMPION: "Champion",
}


def build_group_fixtures(groups: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    """Return the six round-robin pairings for each group."""
    fixtures: dict[str, list[tuple[str, str]]] = {}
    for letter, block in groups.groupby("group"):
        teams = list(block.sort_values("position")["team"])
        pairs = [
            (teams[i], teams[j])
            for i in range(len(teams))
            for j in range(i + 1, len(teams))
        ]
        fixtures[letter] = pairs
    return fixtures


def _third_slots() -> list[tuple[int, int, str]]:
    """Locate the third-placed slots in the R32 template.

    Returns ``(match_index, side_index, reference_group)`` for each slot, sorted
    by reference group so the allocation is deterministic.
    """
    slots = []
    for match_idx, (side_a, side_b) in enumerate(config.R32_TEMPLATE):
        for side_idx, side in enumerate((side_a, side_b)):
            if side[0] == "T":
                slots.append((match_idx, side_idx, side[1]))
    return sorted(slots, key=lambda s: s[2])


class RateTable:
    """Pre-computed neutral-venue scoring rates for a fixed set of teams.

    Caching ``exp(attack_i - defence_j)`` once per simulation batch avoids
    recomputing it for every match of every simulated tournament.
    """

    def __init__(self, model: DixonColesModel, teams: list[str]):
        self.teams = teams
        idx = {t: k for k, t in enumerate(teams)}
        attack = np.array([model.attack.get(t, 0.0) for t in teams])
        defence = np.array([model.defence.get(t, 0.0) for t in teams])
        # rate[i, j] = expected goals for team i against team j.
        self.rate = np.exp(attack[:, None] - defence[None, :])
        self._idx = idx

    def match_rates(self, home: str, away: str) -> tuple[float, float]:
        i, j = self._idx[home], self._idx[away]
        return self.rate[i, j], self.rate[j, i]


def _play_match(home, away, rates, rng, knockout=False):
    """Sample a scoreline; return ``(home_goals, away_goals, winner)``.

    In the group stage ``winner`` is ``None`` on a draw. In the knockout stage a
    draw is broken by a coin flip weighted toward the team that created more
    chances (a lightweight penalty-shootout proxy).
    """
    lam, mu = rates.match_rates(home, away)
    gh = rng.poisson(lam)
    ga = rng.poisson(mu)
    if gh > ga:
        winner = home
    elif ga > gh:
        winner = away
    else:
        winner = None
        if knockout:
            p_home = lam / (lam + mu)
            winner = home if rng.random() < p_home else away
    return gh, ga, winner


def simulate_group(pairs, rates, rng):
    """Simulate one group and return standings ordered best-to-worst.

    Each entry is ``(team, points, goal_difference, goals_for)``. Ties are broken
    by points, goal difference, goals scored, then a random draw standing in for
    head-to-head and the drawing of lots.
    """
    stats = {}
    for home, away in pairs:
        for team in (home, away):
            stats.setdefault(team, {"pts": 0, "gf": 0, "ga": 0})
        gh, ga, winner = _play_match(home, away, rates, rng)
        stats[home]["gf"] += gh
        stats[home]["ga"] += ga
        stats[away]["gf"] += ga
        stats[away]["ga"] += gh
        if winner is None:
            stats[home]["pts"] += config.DRAW_POINTS
            stats[away]["pts"] += config.DRAW_POINTS
        else:
            stats[winner]["pts"] += config.WIN_POINTS

    ordered = sorted(
        stats.items(),
        key=lambda kv: (
            kv[1]["pts"],
            kv[1]["gf"] - kv[1]["ga"],
            kv[1]["gf"],
            rng.random(),
        ),
        reverse=True,
    )
    return [
        (team, s["pts"], s["gf"] - s["ga"], s["gf"]) for team, s in ordered
    ]


def _resolve_r32(group_order, thirds_by_slot):
    """Build the 16 round-of-32 pairings from group results and third allocation."""
    winners = {g: order[0][0] for g, order in group_order.items()}
    runners = {g: order[1][0] for g, order in group_order.items()}

    def resolve(side, match_idx, side_idx):
        kind, ref = side
        if kind == "W":
            return winners[ref]
        if kind == "R":
            return runners[ref]
        return thirds_by_slot[(match_idx, side_idx)]

    matches = []
    for match_idx, (side_a, side_b) in enumerate(config.R32_TEMPLATE):
        matches.append(
            (resolve(side_a, match_idx, 0), resolve(side_b, match_idx, 1))
        )
    return matches


def _play_round(matches, rates, rng, stage, reached):
    """Play a knockout round, record progress, and return the winners in order."""
    winners = []
    for home, away in matches:
        reached[home] = max(reached[home], stage)
        reached[away] = max(reached[away], stage)
        _, _, winner = _play_match(home, away, rates, rng, knockout=True)
        winners.append(winner)
    return winners


def simulate_tournament(fixtures, rates, rng):
    """Play one full tournament; return the champion and each team's deepest stage."""
    reached = {t: GROUP for t in rates.teams}

    group_order = {g: simulate_group(pairs, rates, rng) for g, pairs in fixtures.items()}

    # Best eight third-placed teams, allocated to the template's third slots.
    thirds = [
        (order[2][0], order[2][1], order[2][2], order[2][3], g)
        for g, order in group_order.items()
    ]
    thirds.sort(key=lambda t: (t[1], t[2], t[3], rng.random()), reverse=True)
    qualified = sorted(thirds[: config.N_BEST_THIRDS], key=lambda t: t[4])  # by group letter
    slots = _third_slots()
    thirds_by_slot = {
        (m_idx, s_idx): team
        for (m_idx, s_idx, _ref), (team, *_rest) in zip(slots, qualified)
    }

    r32 = _resolve_r32(group_order, thirds_by_slot)

    r32_winners = _play_round(r32, rates, rng, R32, reached)
    r16 = [(r32_winners[i], r32_winners[j]) for i, j in config.R16_PAIRS]

    r16_winners = _play_round(r16, rates, rng, R16, reached)
    quarters = [(r16_winners[i], r16_winners[j]) for i, j in config.QF_PAIRS]

    qf_winners = _play_round(quarters, rates, rng, QUARTER, reached)
    semis = [(qf_winners[i], qf_winners[j]) for i, j in config.SF_PAIRS]

    sf_winners = _play_round(semis, rates, rng, SEMI, reached)
    final = [(sf_winners[config.FINAL_PAIR[0]], sf_winners[config.FINAL_PAIR[1]])]

    champion = _play_round(final, rates, rng, FINAL, reached)[0]
    reached[champion] = CHAMPION
    return champion, reached


def actual_reached_depths(knockout: pd.DataFrame) -> dict[str, int]:
    """Deepest stage each team actually reached, from the real knockout results.

    Used to back-test the predictions. A team's depth is the deepest round it
    appeared in; the champion is left out because the final may be unplayed on
    the day the data is scraped.
    """
    stage_depth = {
        "Round of 32": R32,
        "Round of 16": R16,
        "Quarter-final": QUARTER,
        "Semi-final": SEMI,
        "Third place": SEMI,  # third-place contestants are the beaten semi-finalists
        "Final": FINAL,
    }
    reached: dict[str, int] = {}
    for _, row in knockout.iterrows():
        depth = stage_depth.get(row["stage"], GROUP)
        for team in (row["home"], row["away"]):
            if isinstance(team, str):
                reached[team] = max(reached.get(team, GROUP), depth)
    return reached


def monte_carlo(
    groups: pd.DataFrame,
    model: DixonColesModel,
    n_sims: int = 10000,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Run ``n_sims`` tournaments and return per-team advancement probabilities.

    Columns: ``team``, ``group`` and the probability of reaching each stage
    (``round_of_32`` ... ``champion``), sorted by title probability.
    """
    rng = np.random.default_rng(seed)
    teams = list(groups["team"])
    rates = RateTable(model, teams)
    fixtures = build_group_fixtures(groups)

    # Count how often each team reaches at least each stage.
    thresholds = [R32, R16, QUARTER, SEMI, FINAL, CHAMPION]
    counts = {t: {s: 0 for s in thresholds} for t in teams}

    for _ in range(n_sims):
        _champion, reached = simulate_tournament(fixtures, rates, rng)
        for team, depth in reached.items():
            for s in thresholds:
                if depth >= s:
                    counts[team][s] += 1

    group_of = dict(zip(groups["team"], groups["group"]))
    rows = []
    for team in teams:
        row = {"team": team, "group": group_of[team]}
        row.update(
            {
                "round_of_32": counts[team][R32] / n_sims,
                "round_of_16": counts[team][R16] / n_sims,
                "quarter_final": counts[team][QUARTER] / n_sims,
                "semi_final": counts[team][SEMI] / n_sims,
                "final": counts[team][FINAL] / n_sims,
                "champion": counts[team][CHAMPION] / n_sims,
            }
        )
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("champion", ascending=False)
        .reset_index(drop=True)
    )
