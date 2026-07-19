"""A time-weighted Dixon-Coles goal model fitted by maximum likelihood.

The model (Dixon & Coles, 1997) gives each team an attack and a defence rating,
adds a global home-advantage term and a low-score dependence parameter ``rho``.
For a match between ``i`` (home) and ``j`` (away) the goal rates are

    lambda = exp(home + attack_i - defence_j)      # home goals
    mu     = exp(       attack_j - defence_i)      # away goals

and the joint probability of the scoreline ``(x, y)`` is

    P(x, y) = tau(x, y; lambda, mu, rho) * Poisson(x; lambda) * Poisson(y; mu)

where ``tau`` inflates/deflates the four low-scoring results (0-0, 1-0, 0-1,
1-1) that an independent double-Poisson model gets wrong.

Two improvements over the "average goals" model this project started from:

* recent tournaments are weighted more heavily (exponential time decay), and
* every team is estimated jointly, so strength is measured relative to the
  quality of the opponents faced, not just raw goal averages.

Teams with no history (World Cup debutants) fall back to league-average ratings
(attack = defence = 0) rather than being treated as unbeatable-by-nobody, which
is what let the old model eliminate them by construction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

# Default exponential time-decay rate, expressed per year. A match played
# ``t`` years before the reference date is weighted by ``exp(-XI_PER_YEAR * t)``;
# 0.05 corresponds to a ~14-year half-life, which keeps a century of World Cups
# relevant while clearly favouring the most recent editions.
XI_PER_YEAR = 0.05

# Ridge (L2) penalty on the attack/defence ratings. World Cup teams have wildly
# uneven match counts (Brazil has played 100+, minnows one or two), so an
# unregularised fit hands extreme ratings to tiny-sample sides. The penalty
# shrinks poorly-observed teams toward the average while barely touching
# well-observed ones. Expressed in the same units as the summed match weights.
RIDGE = 8.0


def _poisson_logpmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Vectorised Poisson log-pmf (avoids a scipy.stats call per element)."""
    return k * np.log(lam) - lam - gammaln(k + 1.0)


def _tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction, evaluated element-wise."""
    out = np.ones_like(lam, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    out[m00] = 1.0 - lam[m00] * mu[m00] * rho
    out[m01] = 1.0 + lam[m01] * rho
    out[m10] = 1.0 + mu[m10] * rho
    out[m11] = 1.0 - rho
    return out


@dataclass
class DixonColesModel:
    """A fitted Dixon-Coles model.

    Ratings are stored as plain dicts so the model round-trips cleanly to JSON.
    """

    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    home_advantage: float = 0.0
    rho: float = 0.0
    teams: list[str] = field(default_factory=list)

    # --- rating lookups ------------------------------------------------------
    def _attack(self, team: str) -> float:
        return self.attack.get(team, 0.0)

    def _defence(self, team: str) -> float:
        return self.defence.get(team, 0.0)

    def expected_goals(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        """Expected goals ``(home, away)``; ``neutral`` drops the home advantage."""
        adv = 0.0 if neutral else self.home_advantage
        lam = np.exp(adv + self._attack(home) - self._defence(away))
        mu = np.exp(self._attack(away) - self._defence(home))
        return float(lam), float(mu)

    # --- match-level probabilities ------------------------------------------
    def scoreline_matrix(
        self, home: str, away: str, max_goals: int = 10, neutral: bool = True
    ) -> np.ndarray:
        """Full ``P(x, y)`` matrix (with the Dixon-Coles correction), rows = home goals."""
        lam, mu = self.expected_goals(home, away, neutral=neutral)
        goals = np.arange(0, max_goals + 1)
        home_p = np.exp(_poisson_logpmf(goals, lam))
        away_p = np.exp(_poisson_logpmf(goals, mu))
        matrix = np.outer(home_p, away_p)

        # Apply the correction to the 2x2 low-score corner.
        corr = np.array(
            [
                [1.0 - lam * mu * self.rho, 1.0 + lam * self.rho],
                [1.0 + mu * self.rho, 1.0 - self.rho],
            ]
        )
        matrix[:2, :2] *= corr
        return matrix / matrix.sum()

    def match_probabilities(
        self, home: str, away: str, max_goals: int = 10, neutral: bool = True
    ) -> tuple[float, float, float]:
        """Return ``(p_home_win, p_draw, p_away_win)``."""
        matrix = self.scoreline_matrix(home, away, max_goals=max_goals, neutral=neutral)
        p_home = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_away = np.triu(matrix, 1).sum()
        return float(p_home), float(p_draw), float(p_away)

    # --- persistence ---------------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        payload = {
            "attack": self.attack,
            "defence": self.defence,
            "home_advantage": self.home_advantage,
            "rho": self.rho,
            "teams": self.teams,
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "DixonColesModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


def _time_weights(years: np.ndarray, reference_year: int, xi: float) -> np.ndarray:
    age = np.maximum(reference_year - years, 0)
    return np.exp(-xi * age)


def fit_dixon_coles(
    matches,
    reference_year: int | None = None,
    xi: float = XI_PER_YEAR,
    ridge: float = RIDGE,
) -> DixonColesModel:
    """Fit a Dixon-Coles model by weighted maximum likelihood.

    ``matches`` must have columns ``HomeTeam``, ``AwayTeam``, ``HomeGoals``,
    ``AwayGoals`` and ``Year``. Attack ratings are constrained to sum to zero
    for identifiability. All matches are treated as played at the home team's
    venue (World Cup fixtures are neutral in practice, but keeping a single
    home-advantage term lets the model learn the small first-listed-team edge in
    the historical data).
    """
    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    index = {t: k for k, t in enumerate(teams)}
    n = len(teams)

    home_idx = matches["HomeTeam"].map(index).to_numpy()
    away_idx = matches["AwayTeam"].map(index).to_numpy()
    x = matches["HomeGoals"].to_numpy(dtype=float)
    y = matches["AwayGoals"].to_numpy(dtype=float)

    if reference_year is None:
        reference_year = int(matches["Year"].max())
    weights = _time_weights(matches["Year"].to_numpy(), reference_year, xi)

    # Parameter vector: n-1 free attacks (attack[0] = -sum of the rest),
    # n defences, home advantage, rho.
    def unpack(params):
        free_attack = params[: n - 1]
        attack = np.empty(n)
        attack[0] = -free_attack.sum()
        attack[1:] = free_attack
        defence = params[n - 1 : 2 * n - 1]
        home_adv = params[2 * n - 1]
        rho = params[2 * n]
        return attack, defence, home_adv, rho

    def neg_log_likelihood(params):
        attack, defence, home_adv, rho = unpack(params)
        lam = np.exp(home_adv + attack[home_idx] - defence[away_idx])
        mu = np.exp(attack[away_idx] - defence[home_idx])
        ll = _poisson_logpmf(x, lam) + _poisson_logpmf(y, mu)
        tau = _tau(x, y, lam, mu, rho)
        ll = ll + np.log(np.clip(tau, 1e-10, None))
        penalty = ridge * (np.sum(attack**2) + np.sum(defence**2))
        return -np.sum(weights * ll) + penalty

    x0 = np.zeros(2 * n + 1)
    x0[2 * n - 1] = 0.25  # a mild positive home advantage to start from
    x0[2 * n] = -0.05  # rho typically comes out slightly negative

    bounds = [(None, None)] * (2 * n - 1) + [(-1.0, 1.0), (-0.2, 0.2)]
    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)

    attack, defence, home_adv, rho = unpack(result.x)
    return DixonColesModel(
        attack={t: float(attack[i]) for t, i in index.items()},
        defence={t: float(defence[i]) for t, i in index.items()},
        home_advantage=float(home_adv),
        rho=float(rho),
        teams=teams,
    )
