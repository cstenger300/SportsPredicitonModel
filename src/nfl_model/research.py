"""Chronological probability calibration using existing out-of-sample forecasts."""
import argparse
import hashlib
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from nfl_model.probability import score_distribution, conditional_win, fit_calibrator, apply_calibrator
from nfl_model.betting import settle


def game_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["game_id", "model"]
    if predictions.duplicated(keys + ["is_home"]).any():
        raise ValueError("Duplicate team predictions")
    home = predictions.loc[predictions.is_home.eq(1)].rename(columns={"points": "actual_home", "predicted_points": "home_mean"})
    away = predictions.loc[predictions.is_home.eq(0)].rename(columns={"points": "actual_away", "predicted_points": "away_mean"})
    result = home.merge(away[keys + ["actual_away", "away_mean"]], on=keys, validate="one_to_one")
    if len(result) * 2 != len(predictions):
        raise ValueError("Unpaired team predictions")
    result["home_residual"] = result.actual_home - result.home_mean
    result["away_residual"] = result.actual_away - result.away_mean
    return result


def raw_observations(games: pd.DataFrame, residuals: dict, schedule: pd.DataFrame) -> pd.DataFrame:
    references = schedule.set_index("game_id")
    rows = []
    for game in games.itertuples():
        distribution = score_distribution(game.model, game.home_mean, game.away_mean, residuals[game.model])
        reference = references.loc[game.game_id] if game.game_id in references.index else None
        markets = [("moneyline", "home", 0.0)]
        if reference is not None and pd.notna(reference.get("spread_line")):
            markets.append(("spread", "home", -float(reference.spread_line)))
        if reference is not None and pd.notna(reference.get("total_line")):
            markets.append(("total", "over", float(reference.total_line)))
        for market, side, point in markets:
            probabilities = distribution.probabilities(market, side, point)
            result = settle(market, side, point, game.actual_home, game.actual_away)
            rows.append({"game_id": game.game_id, "season": game.season, "model": game.model, "market": market,
                         "side": side, "point": point, "base_conditional_win": conditional_win(probabilities),
                         "base_push": probabilities["push"], "result": result,
                         "actual_home": game.actual_home, "actual_away": game.actual_away})
    return pd.DataFrame(rows)


def calibrators_from(observations: pd.DataFrame) -> dict:
    result = {}
    for (kind, market), group in observations.groupby(["model", "market"]):
        decisive = group.loc[group.result.ne("push")]
        result[(kind, market)] = fit_calibrator(decisive.base_conditional_win, decisive.result.eq("win"))
    return result


def prepare_research(prediction_file: Path, schedule_file: Path, output: Path) -> dict:
    predictions = pd.read_csv(prediction_file)
    predictions = predictions.loc[predictions.model.isin(["ridge", "poisson"])]
    games = game_predictions(predictions)
    years = sorted(games.season.unique())
    if len(years) < 3:
        raise ValueError("Need separate residual, calibration, and evaluation seasons")
    first = int(years[0])
    residuals = {kind: group.copy() for kind, group in games.loc[games.season.eq(first)].groupby("model")}
    schedule = pd.read_csv(schedule_file)
    observations = raw_observations(games.loc[games.season.gt(first)], residuals, schedule)
    evaluations = []
    metrics = []
    for season in years[2:]:
        fitting = observations.loc[observations.season.lt(season)]
        calibrators = calibrators_from(fitting)
        test = observations.loc[observations.season.eq(season)].copy()
        for (kind, market), group in test.groupby(["model", "market"]):
            calibrator = calibrators.get((kind, market))
            if calibrator is None:
                continue
            probabilities = []
            for row in group.itertuples():
                raw = {"win": row.base_conditional_win * (1 - row.base_push), "push": row.base_push,
                       "loss": (1 - row.base_conditional_win) * (1 - row.base_push)}
                probabilities.append(conditional_win(apply_calibrator(raw, calibrator)))
            group = group.copy()
            group["calibrated_conditional_win"] = probabilities
            group["calibration_latest_season"] = int(fitting.season.max())
            evaluations.append(group)
            decisive = group.loc[group.result.ne("push")]
            labels = decisive.result.eq("win").astype(int)
            metrics.append({"season": int(season), "model": kind, "market": market,
                            "games": len(group), "pushes": int(group.result.eq("push").sum()),
                            "decisive_games": len(decisive),
                            "raw_brier": brier_score_loss(labels, decisive.base_conditional_win) if len(decisive) else np.nan,
                            "calibrated_brier": brier_score_loss(labels, decisive.calibrated_conditional_win) if len(decisive) else np.nan,
                            "raw_log_loss": log_loss(labels, decisive.base_conditional_win, labels=[0, 1]) if len(decisive) else np.nan,
                            "calibrated_log_loss": log_loss(labels, decisive.calibrated_conditional_win, labels=[0, 1]) if len(decisive) else np.nan})
    if not evaluations:
        raise ValueError("No probability evaluation completed")
    output.mkdir(parents=True, exist_ok=True)
    validation = pd.concat(evaluations, ignore_index=True)
    validation.to_csv(output / "validation.csv", index=False)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(output / "metrics.csv", index=False)
    decisive = validation.loc[validation.result.ne("push")].copy()
    decisive["bin"] = pd.cut(decisive.calibrated_conditional_win, np.linspace(0, 1, 11), include_lowest=True)
    decisive["won"] = decisive.result.eq("win").astype(float)
    reliability = decisive.groupby(["model", "market", "bin"], observed=True).agg(
        games=("won", "size"), predicted=("calibrated_conditional_win", "mean"), actual=("won", "mean")).reset_index()
    reliability.to_csv(output / "reliability.csv", index=False)
    bundle = {"schema_version": 1, "residual_season": first, "residuals": residuals,
              "calibrators": calibrators_from(observations), "observations": observations,
              "calibrated_through_season": int(observations.season.max()),
              "reference": "moneyline outcomes; spread/total reference closing lines without quote timestamps",
              "feature_set": "enhanced",
              "prediction_sha256": hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
              "schedule_sha256": hashlib.sha256(schedule_file.read_bytes()).hexdigest()}
    joblib.dump(bundle, output / "calibration.joblib")
    lines = ["# Chronological probability evaluation", "",
             f"Residual distributions: {first}. Initial calibration: {int(years[1])}. Evaluated seasons: {list(map(int, years[2:]))}.",
             "Each evaluation season's calibrators use only earlier out-of-sample games. The score models were also trained before each evaluated season.",
             "Ridge resamples paired score residuals; Poisson uses independent point counts. Final tie mass is adjusted using the residual season's actual tie rate.",
             "Platt calibration estimates conditional win probabilities separately by model and market; pushes retain their base probability.", "",
             "| Season | Model | Market | Games | Pushes | Raw Brier | Calibrated Brier | Calibrated log loss |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append(f"| {row['season']} | {row['model']} | {row['market']} | {row['games']} | {row['pushes']} | "
                     f"{row['raw_brier']:.4f} | {row['calibrated_brier']:.4f} | {row['calibrated_log_loss']:.4f} |")
    lines += ["", "Lower Brier/log loss is better; Brier/log loss exclude actual pushes and are undefined (NaN) when every outcome is a push. Reliability bins are in reliability.csv.",
              "Spread and total calibration uses reference closing lines. These files have no quote timestamps and do not establish pregame price availability.",
              "No betting profitability or live probability accuracy is established. Enhanced features were already evaluated on these development seasons.",
              "Injuries and weather are not fitted score features. Poisson's NFL scoring assumptions and push probabilities remain research limitations.",
              "Use new prospective data to validate the latest calibration and any recommendation thresholds."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/enhanced-development/predictions.csv"))
    parser.add_argument("--schedule", type=Path, default=Path("data/schedules/games_2026-10-12.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/probability"))
    args = parser.parse_args()
    if not args.schedule.exists():
        import requests
        from nfl_model.prediction import SCHEDULE_URL
        response = requests.get(SCHEDULE_URL, timeout=(15, 60))
        response.raise_for_status()
        args.schedule.parent.mkdir(parents=True, exist_ok=True)
        args.schedule.write_bytes(response.content)
    bundle = prepare_research(args.predictions, args.schedule, args.output_dir)
    print(pd.read_csv(args.output_dir / "metrics.csv").to_string(index=False))
    print(f"Calibration saved through {bundle['calibrated_through_season']}; report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
