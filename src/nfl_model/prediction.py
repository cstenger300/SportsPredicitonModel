"""Pregame point forecasts using only game dates before the requested matchup."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from nfl_model.data import COLUMNS, TEAM_ALIASES, aggregate_games, download_season
from nfl_model.evaluation import make_model
from nfl_model.features import build_features

SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def forecast_game(team_games: pd.DataFrame, game: dict, output: Path | None = None, *, context: dict | None = None) -> dict:
    cutoff = pd.Timestamp(game["gameday"])
    history = team_games.loc[team_games.game_date.lt(cutoff)].copy()
    if history.empty:
        raise ValueError("No historical games before the prediction date")
    home = TEAM_ALIASES.get(game["home_team"], game["home_team"])
    away = TEAM_ALIASES.get(game["away_team"], game["away_team"])
    for team in [home, away]:
        if not history.team.eq(team).any():
            raise ValueError(f"No pregame history for {team}")
    placeholders = []
    for team, opponent, is_home in [(home, away, 1), (away, home, 0)]:
        placeholders.append({"game_id": game["game_id"], "season": int(game["season"]),
                             "week": int(game["week"]), "game_date": cutoff,
                             "team": team, "opponent": opponent, "is_home": is_home,
                             "points": np.nan, "points_allowed": np.nan})
    rows = pd.concat([history, pd.DataFrame(placeholders)], ignore_index=True)
    features, names = build_features(rows, "enhanced")
    train = features.loc[features.season.lt(int(game["season"])) & features.points.notna()]
    upcoming = features.loc[features.game_id.eq(game["game_id"])]
    if train.empty or len(upcoming) != 2:
        raise ValueError("Need prior-season training games and exactly two upcoming team rows")
    results = []
    context = json.loads(json.dumps(context or {}))
    prior_qbs = {team: str(history.loc[history.team.eq(team)].sort_values("game_date").primary_qb_id.iloc[-1])
                 for team in [home, away]}
    context["qb_overrides"] = []
    if context.get("teams"):
        observed = pd.Timestamp(context.get("observed_at"))
        commence = pd.Timestamp(context.get("commence_time"))
        if observed.tzinfo is None or commence.tzinfo is None or observed >= commence:
            raise ValueError("Lineup context must have timezone-aware timestamps before kickoff")
        if observed > pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=60):
            raise ValueError("Lineup context cannot be from the future")
        if commence.tz_convert("America/New_York").date() != cutoff.date():
            raise ValueError("Lineup context date does not match the game")
        for team in [home, away]:
            qb = context["teams"].get(team, {}).get("expected_qb_gsis_id")
            if not qb or qb == prior_qbs[team]:
                continue
            appearances = history.loc[history.primary_qb_id.eq(qb)].sort_values("game_date")
            if appearances.empty:
                context["unmodeled_qb_change"] = True
                continue
            row = appearances.iloc[-1]
            values = [float(appearances.primary_qb_epa.ewm(alpha=0.25, adjust=False).mean().iloc[-1]),
                      float(appearances.primary_qb_dropbacks.sum()), min((cutoff - row.game_date).days, 365)]
            own = upcoming.team.eq(team)
            opposing = upcoming.opponent.eq(team)
            from nfl_model.quarterbacks import QB_FEATURES
            upcoming = upcoming.copy()
            upcoming.loc[own, QB_FEATURES] = values
            upcoming.loc[opposing, [f"opponent_{name}" for name in QB_FEATURES]] = values
            context["qb_overrides"].append({"team": team, "from": prior_qbs[team], "to": qb})
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        upcoming[["game_id", "team"] + names].to_csv(output / "pregame_inputs.csv", index=False)
    for kind in ["ridge", "poisson"]:
        model = make_model(kind)
        model.fit(train[names], train.points)
        predictions = pd.Series(model.predict(upcoming[names]), index=upcoming.team)
        if not np.isfinite(predictions).all():
            raise ValueError("Model produced nonfinite predictions")
        results.append({"model": kind, "away_points": float(predictions[away]),
                        "home_points": float(predictions[home]),
                        "away_margin": float(predictions[away] - predictions[home]),
                        "total_points": float(predictions[away] + predictions[home])})
        if output is not None:
            joblib.dump({"model": model, "features": names, "trained_through_season": int(train.season.max())},
                        output / f"{kind}.joblib")
    return {"game_id": game["game_id"], "gameday": str(cutoff.date()), "away_team": away, "home_team": home,
            "feature_history_before": str(cutoff.date()), "latest_observed_game_date": str(history.game_date.max().date()),
            "trained_through_season": int(train.season.max()), "training_games": int(train.game_id.nunique()),
            "current_season_games_observed": int(history.loc[history.season.eq(game["season"]), "game_id"].nunique()),
            "models": results, "context": context, "previous_primary_qb_ids": prior_qbs,
            "limitations": ["Previous-primary-quarterback proxy unless a timestamped starter override is supplied; overrides are not separately validated.",
                            "Other injuries and forecast weather are displayed context, not fitted score features.",
                            "Probabilities require the separate calibration workflow.",
                            "No verified live odds, expected value, or betting recommendation.",
                            "Historical source data may include revisions after the original games."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Game date, YYYY-MM-DD")
    parser.add_argument("--away", required=True, help="NFL team abbreviation, e.g. BUF")
    parser.add_argument("--home", required=True, help="NFL team abbreviation, e.g. LA")
    parser.add_argument("--start-season", type=int, default=2016)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cutoff = pd.Timestamp(args.date)
    schedule_dir = args.data_dir / "schedules"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    snapshot = schedule_dir / f"games_{cutoff.date()}.csv"
    if not snapshot.exists():
        response = requests.get(SCHEDULE_URL, timeout=(15, 60))
        response.raise_for_status()
        snapshot.write_bytes(response.content)
    schedule = pd.read_csv(snapshot)
    away = TEAM_ALIASES.get(args.away.upper(), args.away.upper())
    home = TEAM_ALIASES.get(args.home.upper(), args.home.upper())
    selected = schedule.loc[schedule.gameday.eq(str(cutoff.date())) & schedule.home_team.replace(TEAM_ALIASES).eq(home)
                            & schedule.away_team.replace(TEAM_ALIASES).eq(away)]
    if len(selected) != 1:
        parser.error("Expected exactly one scheduled matchup for that date and those teams")
    game = selected.iloc[0].to_dict()
    if args.start_season >= int(game["season"]):
        parser.error("start-season must precede the matchup season")
    all_games = []
    manifest = []
    for season in range(args.start_season, int(game["season"]) + 1):
        print(f"Loading {season} ...", flush=True)
        path = download_season(season, args.data_dir / "raw")
        raw = pd.read_parquet(path, columns=COLUMNS)
        # Filter BEFORE aggregation, even if the cache contains future results.
        raw = raw.loc[pd.to_datetime(raw.game_date).lt(cutoff)]
        if not raw.empty:
            all_games.append(aggregate_games(raw))
        manifest.append({"season": season, "file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    output = args.output_dir or Path("artifacts/upcoming") / game["game_id"]
    print("Building time-safe features and training prior-season models ...", flush=True)
    forecast = forecast_game(pd.concat(all_games, ignore_index=True), game, output)
    forecast["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    forecast["schedule_source"] = SCHEDULE_URL
    forecast["schedule_snapshot"] = str(snapshot)
    forecast["schedule_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    (output / "prediction.json").write_text(json.dumps(forecast, indent=2) + "\n")
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(forecast, indent=2))
    print(f"Forecast saved to {output / 'prediction.json'}")


if __name__ == "__main__":
    main()
