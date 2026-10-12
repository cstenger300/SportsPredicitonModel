"""Command-line entry point for the historical research workflow."""
import argparse
from pathlib import Path
import json
import hashlib
import pandas as pd
from nfl_model.data import download_season, aggregate_games, COLUMNS
from nfl_model.features import build_features
from nfl_model.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2016)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--first-test-season", type=int, default=2021)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--feature-set", choices=["baseline", "enhanced"], default="baseline")
    parser.add_argument("--development-only", action="store_true",
                        help="Label all evaluated seasons as development, not an untouched holdout")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path("artifacts") if args.feature_set == "baseline" else Path("artifacts/enhanced")
    if not args.start_season < args.first_test_season <= args.holdout_season <= args.end_season:
        parser.error("Require start < first test <= holdout <= end season")
    all_games = []
    manifest = []
    for season in range(args.start_season, args.end_season + 1):
        print(f"Loading {season} ...", flush=True)
        path = download_season(season, args.data_dir / "raw")
        manifest.append({"season": season, "file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        games = aggregate_games(pd.read_parquet(path, columns=COLUMNS))
        print(f"  {games.game_id.nunique()} regular-season games", flush=True)
        all_games.append(games)
    print(f"Building {args.feature_set} features ...", flush=True)
    features, names = build_features(pd.concat(all_games, ignore_index=True), args.feature_set)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    feature_file = f"pregame_features_{args.feature_set}_{args.start_season}_{args.end_season}.parquet"
    features.to_parquet(args.data_dir / feature_file, index=False)
    metrics = evaluate(features, names, args.first_test_season, args.holdout_season, args.output_dir,
                       feature_set=args.feature_set, final_holdout=not args.development_only)
    (args.output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "run_config.json").write_text(json.dumps({
        "start_season": args.start_season, "end_season": args.end_season,
        "first_test_season": args.first_test_season, "last_test_season": args.holdout_season,
        "feature_set": args.feature_set, "development_only": args.development_only,
        "features_file": str(args.data_dir / feature_file),
    }, indent=2) + "\n")
    print(metrics.to_string(index=False))
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
