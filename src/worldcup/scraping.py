"""Scrape World Cup match data from Wikipedia.

Two things are collected here:

* Historical match results (``home``, ``score``, ``away``, ``year``) used to fit
  the goal model. These come from the per-edition tournament articles, which
  embed each match as a ``div.footballbox``.
* The 2026 group standings and the actual knockout results, used to build the
  fixture to simulate and to back-test the predictions.

Note on coverage: recent editions (2018 onwards) split their group-stage
matches into separate sub-articles whose markup is inconsistent, so the
``footballbox`` scraper only reliably returns the knockout matches for those
years. The historical training set is therefore shipped as a versioned raw
snapshot in ``data/raw`` (1930-2018); :func:`scrape_edition` remains available
for transparency and to refresh older editions.
"""
from __future__ import annotations

import io
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

WIKI = "https://en.wikipedia.org/wiki/"
# Wikipedia rejects the default urllib/requests user agent, so identify the bot.
HEADERS = {
    "User-Agent": (
        "worldcup-predictor/0.1 (portfolio project; "
        "https://github.com/; contact via GitHub issues)"
    )
}
REQUEST_TIMEOUT = 30


def _get_html(url: str) -> str:
    """Fetch a URL and return its HTML, raising on any HTTP error."""
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def _parse_footballboxes(html: str) -> pd.DataFrame:
    """Extract every ``div.footballbox`` match from a page as home/score/away rows."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for box in soup.find_all("div", class_="footballbox"):
        home = box.find("th", class_="fhome")
        score = box.find("th", class_="fscore")
        away = box.find("th", class_="faway")
        if home and score and away:
            rows.append(
                {
                    "home": home.get_text(),
                    "score": score.get_text(),
                    "away": away.get_text(),
                }
            )
    return pd.DataFrame(rows)


def scrape_edition(year: int) -> pd.DataFrame:
    """Scrape the match results of a single World Cup edition.

    Returns a frame with columns ``home``, ``score``, ``away``, ``year``. For
    older editions this is the full schedule; for recent editions it may only
    cover the matches still embedded on the main article (see module docstring).
    """
    html = _get_html(f"{WIKI}{year}_FIFA_World_Cup")
    df = _parse_footballboxes(html)
    df["year"] = year
    return df


def scrape_historical(years: list[int], pause: float = 1.0) -> pd.DataFrame:
    """Scrape and concatenate several editions, pausing between requests."""
    frames = []
    for year in years:
        frames.append(scrape_edition(year))
        time.sleep(pause)
    return pd.concat(frames, ignore_index=True)


# --- 2026 tournament ---------------------------------------------------------

def scrape_2026_groups() -> pd.DataFrame:
    """Scrape the twelve 2026 group standings.

    Returns one row per team with columns ``group``, ``position`` (1-4) and
    ``team`` (still raw, cleaned downstream). The standings tables live at even
    ``read_html`` indices 10..32 on the main article.
    """
    from .config import GROUP_LETTERS_2026

    html = _get_html(f"{WIKI}2026_FIFA_World_Cup")
    tables = pd.read_html(io.StringIO(html))
    rows = []
    for letter, idx in zip(GROUP_LETTERS_2026, range(10, 33, 2)):
        table = tables[idx]
        team_col = next(c for c in table.columns if str(c).startswith("Team"))
        for position, team in zip(table["Pos"], table[team_col]):
            rows.append({"group": letter, "position": int(position), "team": team})
    return pd.DataFrame(rows)


def scrape_2026_knockout() -> pd.DataFrame:
    """Scrape the actual 2026 knockout results (used only for back-testing).

    The knockout sub-article lists its 32 matches as ``footballbox`` divs in
    bracket order: 16 round-of-32, 8 round-of-16, 4 quarter-finals, 2 semis,
    the third-place play-off and the final.
    """
    html = _get_html(f"{WIKI}2026_FIFA_World_Cup_knockout_stage")
    return _parse_footballboxes(html)
