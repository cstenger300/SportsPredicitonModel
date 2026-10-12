"""Repeatable refresh, immutable snapshots, research pricing, and optional paper logging."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading

import joblib
import pandas as pd
import requests

from nfl_model.betting import QUOTE_COLUMNS, validate_quotes, price_quotes
from nfl_model.data import COLUMNS, aggregate_games, download_season, TEAM_ALIASES
from nfl_model.paper import PaperLedger
from nfl_model.prediction import SCHEDULE_URL, forecast_game
from nfl_model.probability import score_distribution, apply_calibrator
from nfl_model.sources import SourceUnavailable, fetch_lineups, fetch_weather, fetch_odds, map_quarterback_ids, load_roster

ROOT = Path(__file__).resolve().parents[2]


def select_game(schedule, date, away, home):
    away = TEAM_ALIASES.get(away.upper(), away.upper())
    home = TEAM_ALIASES.get(home.upper(), home.upper())
    matches = schedule.loc[schedule.gameday.eq(date) & schedule.away_team.replace(TEAM_ALIASES).eq(away)
                           & schedule.home_team.replace(TEAM_ALIASES).eq(home)]
    if len(matches) != 1:
        raise ValueError("Expected one scheduled matchup")
    game = matches.iloc[0].to_dict()
    game["home_team"], game["away_team"] = home, away
    return game


def update(date: str, away: str, home: str, *, root=ROOT, refresh=False, context_file=None, quotes_file=None, paper=False):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    status = {"odds_key_configured": bool(os.environ.get("ODDS_API_KEY"))}
    schedule_dir = root / "data/schedules"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    cached = schedule_dir / f"games_{date}.csv"
    if refresh or not cached.exists():
        try:
            response = requests.get(SCHEDULE_URL, timeout=(15, 60))
            response.raise_for_status()
        except requests.RequestException:
            raise SourceUnavailable("Schedule refresh unavailable") from None
        snapshot = schedule_dir / f"{stamp}_games.csv"
        snapshot.write_bytes(response.content)
        schedule_path = snapshot
    else:
        schedule_path = cached
    schedule = pd.read_csv(schedule_path)
    game = select_game(schedule, date, away, home)
    run_dir = root / "artifacts/upcoming" / game["game_id"] / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshots = run_dir / "snapshots"
    if context_file:
        context = json.loads(Path(context_file).read_text())
        status["lineups"] = "user-supplied timestamped context"
    else:
        try:
            context = fetch_lineups(game, snapshots)
            status["lineups"] = "depth chart/injuries fetched; starter confirmation requires review"
        except SourceUnavailable as error:
            context = {"teams": {}, "lineups_reviewed": False}
            status["lineups"] = str(error)
    kickoff = pd.Timestamp(f"{game['gameday']} {game['gametime']}").tz_localize("America/New_York").tz_convert("UTC")
    context.setdefault("commence_time", kickoff.isoformat())
    context.setdefault("observed_at", datetime.now(timezone.utc).isoformat())
    try:
        roster = load_roster(int(game["season"]), root / "data", refresh=refresh)
        context = map_quarterback_ids(context, roster)
        status["roster"] = "quarterback ID mapping loaded"
    except SourceUnavailable as error:
        status["roster"] = str(error)
    if "weather" not in context:
        try:
            context["weather"] = fetch_weather(game, context, snapshots)
            status["weather"] = "forecast fetched; displayed, not fitted in score models"
        except SourceUnavailable as error:
            status["weather"] = str(error)
    else:
        status["weather"] = "user-supplied context; displayed, not fitted in score models"
    cutoff = pd.Timestamp(date)
    games = []
    manifest = []
    for season in range(2016, int(game["season"]) + 1):
        path = download_season(season, root / "data/raw", refresh=refresh and season == int(game["season"]))
        raw = pd.read_parquet(path, columns=COLUMNS)
        raw = raw.loc[pd.to_datetime(raw.game_date).lt(cutoff)]
        if not raw.empty:
            games.append(aggregate_games(raw))
        manifest.append({"season": season, "file": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    forecast = forecast_game(pd.concat(games, ignore_index=True), game, run_dir, context=context)
    forecast["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    forecast["schedule_source"] = SCHEDULE_URL
    forecast["schedule_sha256"] = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    quotes = pd.DataFrame(columns=QUOTE_COLUMNS)
    if quotes_file:
        quotes = validate_quotes(pd.read_csv(quotes_file))
        status["odds"] = "timestamped CSV import; provider authenticity depends on supplied source"
    else:
        try:
            quotes = fetch_odds(game, snapshots)
            status["odds"] = "live prices fetched"
        except SourceUnavailable as error:
            status["odds"] = str(error)
    ledger = PaperLedger(root / "data/paper.sqlite")
    ledger.record_quotes(quotes)
    ledger.settle_from_schedule(schedule)
    priced = pd.DataFrame()
    calibration_path = root / "artifacts/probability/calibration.joblib"
    if calibration_path.exists():
        bundle = joblib.load(calibration_path)
    else:
        bundle = None
    if bundle is not None and int(bundle["calibrated_through_season"]) < int(game["season"]):
        win_estimates = []
        for model in forecast["models"]:
            kind = model["model"]
            distribution = score_distribution(kind, model["home_points"], model["away_points"], bundle["residuals"][kind])
            win_estimates.append(apply_calibrator(distribution.probabilities("moneyline", "home"),
                                                 bundle["calibrators"].get((kind, "moneyline"))))
        forecast["probabilities"] = {"home_win": sum(p["win"] for p in win_estimates) / len(win_estimates),
                                     "away_win": sum(p["loss"] for p in win_estimates) / len(win_estimates),
                                     "tie": sum(p["push"] for p in win_estimates) / len(win_estimates),
                                     "calibrated_through_season": bundle["calibrated_through_season"],
                                     "status": "research probability estimates; not prospectively validated"}
        priced = price_quotes(forecast, quotes, bundle)
        status["calibration"] = "prior-season calibration applied"
    else:
        status["calibration"] = "no calibration fitted strictly before this game's season"
    recorded = False
    if paper and not priced.empty:
        candidates = priced.loc[priced.research_candidate]
        if not candidates.empty:
            recorded = ledger.record_bet(candidates.iloc[0].to_dict(), forecast)
    forecast["source_status"] = status
    (run_dir / "prediction.json").write_text(json.dumps(forecast, indent=2) + "\n")
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    quotes.to_csv(run_dir / "quotes.csv", index=False)
    priced.to_csv(run_dir / "priced_quotes.csv", index=False)
    state = {"game_id": game["game_id"], "date": date, "away": game["away_team"], "home": game["home_team"],
             "commence_time": context["commence_time"],
             "run_dir": str(run_dir.relative_to(root)), "generated_at_utc": forecast["generated_at_utc"],
             "source_status": status, "quotes": len(quotes), "paper_bet_recorded": recorded, "paper_summary": ledger.summary()}
    dashboard = root / "artifacts/dashboard"
    dashboard.mkdir(exist_ok=True)
    temporary = dashboard / "state.part"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(dashboard / "state.json")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--home", required=True)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--context-json", type=Path)
    parser.add_argument("--odds-csv", type=Path)
    parser.add_argument("--paper", action="store_true", help="Log at most one eligible research paper bet per game")
    parser.add_argument("--watch-seconds", type=int, default=0, help="Repeat at this interval; minimum 300 seconds")
    args = parser.parse_args()
    if args.watch_seconds and args.watch_seconds < 300:
        parser.error("Watch interval must be at least 300 seconds")
    stop = threading.Event()
    while True:
        state = None
        try:
            state = update(args.date, args.away, args.home, refresh=args.refresh,
                           context_file=args.context_json, quotes_file=args.odds_csv, paper=args.paper)
            print(json.dumps(state, indent=2), flush=True)
        except (SourceUnavailable, ValueError) as error:
            print(json.dumps({"error": str(error)}), flush=True)
            if not args.watch_seconds:
                raise SystemExit(1) from None
        if not args.watch_seconds:
            break
        if state is not None and not state["source_status"]["odds_key_configured"]:
            print("Watcher stopped: configure ODDS_API_KEY before scheduling live odds updates.", flush=True)
            raise SystemExit(1)
        if state is not None and pd.Timestamp.now(tz="UTC") >= pd.Timestamp(state["commence_time"]):
            print("Watcher stopped: kickoff has passed. Run a later refresh to settle paper bets.", flush=True)
            break
        try:
            stop.wait(args.watch_seconds)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
