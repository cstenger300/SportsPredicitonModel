"""Compare feature sets on identical development games and source inputs."""
import argparse
import json
from pathlib import Path
import pandas as pd


def compare(baseline_dir: Path, enhanced_dir: Path, output: Path):
    manifests = []
    frames = []
    for directory in [baseline_dir, enhanced_dir]:
        manifest = json.loads((directory / "data_manifest.json").read_text())
        manifests.append({entry["season"]: entry["sha256"] for entry in manifest})
        metrics = pd.read_csv(directory / "metrics.csv")
        if not metrics.split.eq("development").all():
            raise ValueError("Feature comparison requires development-only reports")
        predictions = pd.read_csv(directory / "predictions.csv")
        keys = ["season", "game_id", "team", "model"]
        if predictions.duplicated(keys).any():
            raise ValueError("Duplicate prediction records")
        frames.append(predictions.sort_values(keys).reset_index(drop=True))
    if manifests[0] != manifests[1]:
        raise ValueError("Feature sets must use identical historical source files")
    key_and_outcome = ["season", "game_id", "team", "opponent", "model", "is_home", "points"]
    if not frames[0][key_and_outcome].equals(frames[1][key_and_outcome]):
        raise ValueError("Feature sets must predict identical games and outcomes")
    results = []
    for label, frame in zip(["baseline", "enhanced"], frames):
        for kind in ["ridge", "poisson"]:
            scored = frame.loc[frame.model.eq(kind)]
            home = scored.loc[scored.is_home.eq(1)].set_index("game_id")
            away = scored.loc[scored.is_home.eq(0)].set_index("game_id").reindex(home.index)
            margin = home.points - away.points
            predicted_margin = home.predicted_points - away.predicted_points
            total = home.points + away.points
            predicted_total = home.predicted_points + away.predicted_points
            decided = margin.ne(0)
            results.append({"features": label, "model": kind, "games": len(home),
                            "team_points_mae": float((scored.points - scored.predicted_points).abs().mean()),
                            "margin_mae": float((margin - predicted_margin).abs().mean()),
                            "total_mae": float((total - predicted_total).abs().mean()),
                            "winner_accuracy": float((margin[decided].gt(0) == predicted_margin[decided].gt(0)).mean())})
    summary = pd.DataFrame(results)
    lines = ["# Feature comparison: development seasons", "",
             f"Seasons: {sorted(frames[0].season.unique().tolist())}. Identical source hashes, games, and outcomes verified.",
             "Both versions use the same franchise normalization. Model hyperparameters are unchanged.", "",
             "| Features | Model | Games | Score MAE | Margin MAE | Total MAE | Winner accuracy |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['features']} | {row['model']} | {row['games']} | {row['team_points_mae']:.2f} | "
                     f"{row['margin_mae']:.2f} | {row['total_mae']:.2f} | {row['winner_accuracy']:.1%} |")
    lines += ["", "These are pooled development results, not an untouched final test or a betting backtest.",
              "Enhanced adds opponent-adjusted EPA and previous-primary-quarterback history. The additions are tested together; this does not isolate their individual effects.",
              "Check the separate season reports for variability. No odds, calibrated probabilities, or profitable-betting claim is included."]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    summary.to_csv(output.with_suffix(".csv"), index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baseline-development"))
    parser.add_argument("--enhanced-dir", type=Path, default=Path("artifacts/enhanced-development"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/feature_comparison.md"))
    args = parser.parse_args()
    print(compare(args.baseline_dir, args.enhanced_dir, args.output).to_string(index=False))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
