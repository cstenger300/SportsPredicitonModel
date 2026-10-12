"""Fetch one live or historical NFL odds snapshot without logging credentials."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from nfl_model.sources import fetch_odds, SourceUnavailable
from nfl_model.workflow import ROOT, select_game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--home", required=True)
    parser.add_argument("--historical-at", help="Timezone-aware pregame snapshot timestamp; paid historical access may be required")
    parser.add_argument("--schedule", type=Path)
    args = parser.parse_args()
    schedule_file = args.schedule or ROOT / "data/schedules" / f"games_{args.date}.csv"
    if not schedule_file.exists():
        available = list((ROOT / "data/schedules").glob("*.csv"))
        if not available:
            parser.error("No schedule snapshot found; run nfl-predict first")
        schedule_file = max(available, key=lambda p: p.stat().st_mtime)
    game = select_game(pd.read_csv(schedule_file), args.date, args.away, args.home)
    directory = ROOT / "data/odds" / game["game_id"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        quotes = fetch_odds(game, directory / "snapshots", historical_at=args.historical_at)
    except SourceUnavailable as error:
        parser.exit(1, str(error) + "\n")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{stamp}_quotes.csv"
    quotes.to_csv(output, index=False)
    print(f"Saved {len(quotes)} timestamped prices to {output}")


if __name__ == "__main__":
    main()
