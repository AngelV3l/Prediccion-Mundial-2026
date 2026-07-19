"""Scrape the 2026 group standings and knockout results into data/raw.

The historical training snapshot (1930-2018) is versioned in the repository, so
it is only checked for here. Pass --refresh-historical to re-scrape it from
Wikipedia (see scraping.scrape_edition for coverage caveats on recent editions).

Usage:
    python scripts/01_scrape_data.py
    python scripts/01_scrape_data.py --refresh-historical
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldcup import config
from worldcup import scraping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-historical",
        action="store_true",
        help="Re-scrape the historical results instead of using the versioned snapshot.",
    )
    args = parser.parse_args()

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.refresh_historical:
        print("Scraping historical results (this hits Wikipedia once per edition)...")
        historical = scraping.scrape_historical(config.WORLD_CUP_YEARS)
        historical.to_csv(config.RAW_HISTORICAL, index=False)
        print(f"  wrote {len(historical)} matches -> {config.RAW_HISTORICAL}")
    elif not config.RAW_HISTORICAL.exists():
        raise SystemExit(
            f"Missing historical snapshot at {config.RAW_HISTORICAL}. "
            "Run with --refresh-historical to build it."
        )
    else:
        print(f"Using versioned historical snapshot at {config.RAW_HISTORICAL}")

    print("Scraping 2026 group standings...")
    groups = scraping.scrape_2026_groups()
    groups.to_csv(config.RAW_GROUPS_2026, index=False)
    print(f"  wrote {len(groups)} team rows -> {config.RAW_GROUPS_2026}")

    print("Scraping 2026 knockout results (for back-testing)...")
    knockout = scraping.scrape_2026_knockout()
    knockout.to_csv(config.RAW_FIXTURE_2026, index=False)
    print(f"  wrote {len(knockout)} matches -> {config.RAW_FIXTURE_2026}")


if __name__ == "__main__":
    main()
