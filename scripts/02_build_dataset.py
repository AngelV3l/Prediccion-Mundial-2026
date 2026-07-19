"""Clean the raw snapshots in data/raw into model-ready files in data/processed.

Usage:
    python scripts/02_build_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldcup import cleaning, config


def main() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    historical = cleaning.clean_historical(pd.read_csv(config.RAW_HISTORICAL))
    historical.to_csv(config.CLEAN_HISTORICAL, index=False)
    print(
        f"Historical: {len(historical)} matches, {historical['Year'].min()}-"
        f"{historical['Year'].max()} -> {config.CLEAN_HISTORICAL}"
    )

    groups = cleaning.clean_2026_groups(pd.read_csv(config.RAW_GROUPS_2026))
    groups.to_csv(config.CLEAN_GROUPS_2026, index=False)
    print(f"2026 groups: {len(groups)} teams -> {config.CLEAN_GROUPS_2026}")

    knockout = cleaning.clean_2026_knockout(pd.read_csv(config.RAW_FIXTURE_2026))
    knockout.to_csv(config.CLEAN_FIXTURE_2026, index=False)
    print(f"2026 knockout: {len(knockout)} matches -> {config.CLEAN_FIXTURE_2026}")


if __name__ == "__main__":
    main()
