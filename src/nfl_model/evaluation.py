"""Walk-forward evaluation. Hyperparameters fixed before the held-out season."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def make_model(kind: str):
    estimator = Ridge(alpha=20.0) if kind == "ridge" else PoissonRegressor(alpha=0.1, max_iter=1000)
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), estimator)


def evaluate(frame: pd.DataFrame, features: list[str], first_test: int, holdout: int, output: Path,
             *, feature_set: str = "baseline", final_holdout: bool = True):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    season_metrics = []
    available = set(frame.season.unique())
    for season in range(first_test, holdout + 1):
        train = frame.loc[frame.season.lt(season)]
        test = frame.loc[frame.season.eq(season)]
        if season not in available or train.empty:
            raise ValueError(f"Missing training data or evaluation season {season}")
        for kind in ["baseline", "ridge", "poisson"]:
            if kind == "baseline":
                # Historical average team score, separately for home and away teams.
                means = train.groupby("is_home").points.mean()
                predictions = test.is_home.map(means).to_numpy()
            else:
                model = make_model(kind)
                model.fit(train[features], train.points)
                predictions = model.predict(test[features])
                joblib.dump({"model": model, "features": features, "trained_before_season": season},
                            output / f"{kind}_{season}.joblib")
            scored = test[["game_id", "season", "week", "game_date", "team", "opponent", "is_home", "points"]].copy()
            scored["predicted_points"] = predictions
            scored["model"] = kind
            results.append(scored)
            home = scored.loc[scored.is_home.eq(1)].set_index("game_id")
            away = scored.loc[scored.is_home.eq(0)].set_index("game_id").reindex(home.index)
            actual_margin = home.points - away.points
            predicted_margin = home.predicted_points - away.predicted_points
            actual_total = home.points + away.points
            predicted_total = home.predicted_points + away.predicted_points
            decided = actual_margin.ne(0)
            season_metrics.append({
                "season": season, "split": "holdout" if final_holdout and season == holdout else "development",
                "model": kind, "training_games": int(train.game_id.nunique()), "games": len(home),
                "team_points_mae": mean_absolute_error(test.points, predictions),
                "team_points_rmse": float(np.sqrt(mean_squared_error(test.points, predictions))),
                "margin_mae": mean_absolute_error(actual_margin, predicted_margin),
                "total_mae": mean_absolute_error(actual_total, predicted_total),
                "winner_accuracy": float(((actual_margin[decided] > 0) == (predicted_margin[decided] > 0)).mean()),
                "tied_games_excluded_from_winner_accuracy": int((~decided).sum()),
            })
    pd.concat(results, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    metrics = pd.DataFrame(season_metrics)
    metrics.to_csv(output / "metrics.csv", index=False)
    (output / "feature_names.json").write_text(json.dumps(features, indent=2) + "\n")
    lines = ["# NFL chronological baseline evaluation", "",
             (f"Final holdout season: {holdout}." if final_holdout else "Development evaluation only; no final holdout was evaluated."),
             f"Feature set: {feature_set}. Each model is trained only on seasons before its evaluation season.",
             "Features update from completed prior games during the evaluation season; models refit between seasons only.",
             "Hyperparameters are fixed defaults, not selected against the holdout.", "",
             "| Season | Split | Model | Games | Score MAE | Margin MAE | Total MAE | Winner accuracy |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in season_metrics:
        lines.append(f"| {row['season']} | {row['split']} | {row['model']} | {row['games']} | "
                     f"{row['team_points_mae']:.2f} | {row['margin_mae']:.2f} | {row['total_mae']:.2f} | {row['winner_accuracy']:.1%} |")
    lines += ["", "## Interpretation and limitations", "",
              "MAE is measured in points; lower is better. Winner accuracy excludes actual ties; predicted ties select away.",
              "These are research baselines, not validated betting recommendations. No odds, ROI, calibrated probabilities, or player props are claimed.",
              "Basic Poisson regression assumes equidispersed integer points, which does not represent NFL touchdown/field-goal scoring well.",
              "Pace is represented by plays per game, not neutral-situation pace.",
              ("Enhanced inputs include weekly opponent-adjusted EPA ratings and the previous game's primary-quarterback history. "
               "The quarterback proxy is not a confirmed or expected upcoming starter."
               if feature_set == "enhanced" else "Baseline features do not include opponent adjustments or quarterback-history proxies."),
              "Timestamped quarterback availability, injuries, weather, and betting lines are not yet included.",
              "Historical nflverse data can be revised and is not a point-in-time archive of what was published before kickoff.",
              "If these results guide subsequent changes, the holdout becomes development evidence; use a new untouched period for final validation."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    # Separate deployable research models from the strictly historical evaluation models.
    history = frame.loc[frame.season.le(holdout)]
    for kind in ["ridge", "poisson"]:
        model = make_model(kind)
        model.fit(history[features], history.points)
        joblib.dump({"model": model, "features": features, "trained_through_season": holdout},
                    output / f"{kind}_latest.joblib")
    return metrics
