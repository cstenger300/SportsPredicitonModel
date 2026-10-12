"""Evaluate one flat-unit research bet per game; timestamped odds required by default."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from nfl_model.betting import validate_quotes, expected_value, settle, unit_profit
from nfl_model.probability import score_distribution, apply_calibrator
from nfl_model.research import game_predictions, calibrators_from


def closing_reference_quotes(schedule: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game in schedule.itertuples():
        values = [("moneyline", "home", 0.0, game.home_moneyline),
                  ("moneyline", "away", 0.0, game.away_moneyline),
                  ("spread", "home", -game.spread_line, game.home_spread_odds),
                  ("spread", "away", game.spread_line, game.away_spread_odds),
                  ("total", "over", game.total_line, game.over_odds),
                  ("total", "under", game.total_line, game.under_odds)]
        for market, side, point, price in values:
            if pd.notna(point) and pd.notna(price) and abs(price) >= 100:
                rows.append({"game_id": game.game_id, "bookmaker": "nflverse-reference", "market": market,
                             "side": side, "point": point, "american_odds": price, "source": "closing-reference-no-timestamps"})
    return pd.DataFrame(rows)


def backtest(predictions: pd.DataFrame, quotes: pd.DataFrame, bundle: dict, output: Path,
             *, closing_reference=False, decision_minutes=60) -> dict:
    games = game_predictions(predictions.loc[predictions.model.isin(["ridge", "poisson"])])
    input_quotes = len(quotes)
    if not closing_reference:
        quotes = validate_quotes(quotes)
        quotes = quotes.loc[quotes.observed_at.le(quotes.commence_time - pd.Timedelta(minutes=decision_minutes))].copy()
        quotes = quotes.loc[quotes.observed_at.ge(quotes.commence_time - pd.Timedelta(minutes=decision_minutes + 15)) &
                            quotes.source_updated_at.ge(quotes.commence_time - pd.Timedelta(minutes=decision_minutes + 15))]
        quotes = quotes.sort_values("observed_at").drop_duplicates(["game_id", "bookmaker", "market", "side"], keep="last")
    records = []
    evaluated_games = 0
    for game_id, group in games.groupby("game_id", sort=True):
        season = int(group.season.iloc[0])
        earlier = bundle["observations"].loc[bundle["observations"].season.lt(season)]
        if earlier.empty or season <= int(bundle["residual_season"]) + 1:
            continue
        calibrators = calibrators_from(earlier)
        candidates = quotes.loc[quotes.game_id.eq(game_id)]
        if candidates.empty:
            continue
        evaluated_games += 1
        distributions = {g.model: score_distribution(g.model, g.home_mean, g.away_mean, bundle["residuals"][g.model])
                         for g in group.itertuples()}
        priced = []
        for quote in candidates.itertuples():
            estimates = []
            for kind, distribution in distributions.items():
                calibrator = calibrators.get((kind, quote.market))
                if calibrator is None:
                    continue
                raw = distribution.probabilities(quote.market, quote.side, quote.point)
                estimates.append(apply_calibrator(raw, calibrator, reverse=quote.side in {"away", "under"}))
            if len(estimates) != 2:
                continue
            average = {key: float(np.mean([e[key] for e in estimates])) for key in ["win", "push", "loss"]}
            ev = expected_value(average, quote.american_odds)
            if ev < 0.03 or min(expected_value(e, quote.american_odds) for e in estimates) <= 0:
                continue
            priced.append({**quote._asdict(), "estimated_ev": ev, "p_win": average["win"], "p_push": average["push"]})
        if not priced:
            continue
        selected = sorted(priced, key=lambda row: (-row["estimated_ev"], row["bookmaker"], row["market"], row["side"]))[0]
        actual = group.iloc[0]
        result = settle(selected["market"], selected["side"], selected["point"], actual.actual_home, actual.actual_away)
        records.append({**selected, "season": season, "game_date": actual.game_date,
                        "result": result, "profit_units": unit_profit(result, selected["american_odds"]),
                        "calibration_latest_season": int(earlier.season.max())})
    output.mkdir(parents=True, exist_ok=True)
    bets = pd.DataFrame(records)
    if not bets.empty:
        bets = bets.sort_values(["game_date", "game_id"])
        bets["cumulative_profit_units"] = bets.profit_units.cumsum()
        drawdown = float((bets.cumulative_profit_units.cummax().clip(lower=0) - bets.cumulative_profit_units).max())
    else:
        drawdown = 0.0
    bets.to_csv(output / "bets.csv", index=False)
    summary = {"mode": "closing-reference diagnostic" if closing_reference else "timestamped historical odds",
               "input_quotes": input_quotes, "quotes_after_cutoff": len(quotes), "games_with_quotes": evaluated_games,
               "bets": len(bets), "wins": int(bets.result.eq("win").sum()) if len(bets) else 0,
               "losses": int(bets.result.eq("loss").sum()) if len(bets) else 0,
               "pushes": int(bets.result.eq("push").sum()) if len(bets) else 0,
               "profit_units": float(bets.profit_units.sum()) if len(bets) else 0.0,
               "roi": float(bets.profit_units.mean()) if len(bets) else None,
               "max_drawdown_units": drawdown, "decision_minutes_before_kickoff": None if closing_reference else decision_minutes,
               "stakes": "One flat unit per selected game, including pushes; maximum one bet per game",
               "validation_limitations": ["Research score features lack timestamped injury/weather history.",
                                          "Development seasons have already influenced feature choices.",
                                          "nflverse score data can contain retrospective corrections."]}
    if closing_reference:
        summary["validation_limitations"].append("Closing reference lines have no observation timestamps; pregame availability and actionable returns are NOT established.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "report.md").write_text("# " + summary["mode"] + "\n\n" +
        "This is a development experiment, not proof of profitable live betting.\n\n" +
        "```json\n" + json.dumps(summary, indent=2) + "\n```\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/enhanced-development/predictions.csv"))
    parser.add_argument("--calibration", type=Path, default=Path("artifacts/probability/calibration.joblib"))
    parser.add_argument("--odds-csv", type=Path)
    parser.add_argument("--closing-reference", action="store_true")
    parser.add_argument("--schedule", type=Path, default=Path("data/schedules/games_2026-10-12.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/backtest"))
    parser.add_argument("--decision-minutes", type=int, default=60)
    args = parser.parse_args()
    if args.closing_reference == bool(args.odds_csv):
        parser.error("Choose --odds-csv OR --closing-reference explicitly")
    if args.decision_minutes < 0:
        parser.error("decision-minutes cannot be negative")
    quotes = closing_reference_quotes(pd.read_csv(args.schedule)) if args.closing_reference else pd.read_csv(args.odds_csv)
    bundle = joblib.load(args.calibration)
    print(json.dumps(backtest(pd.read_csv(args.predictions), quotes, bundle, args.output_dir,
                             closing_reference=args.closing_reference, decision_minutes=args.decision_minutes), indent=2))


if __name__ == "__main__":
    main()
