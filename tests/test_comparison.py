import json
import pandas as pd
import pytest
from nfl_model.comparison import compare


def reports(tmp_path):
    directories = [tmp_path / "baseline", tmp_path / "enhanced"]
    rows = [{"season": 2021, "game_id": "g1", "team": "A" if home else "B",
             "opponent": "B" if home else "A", "model": model, "is_home": home,
             "points": 21 if home else 14, "predicted_points": 20 if home else 15}
            for home in [0, 1] for model in ["baseline", "ridge", "poisson"]]
    for directory in directories:
        directory.mkdir()
        (directory / "data_manifest.json").write_text(json.dumps([{"season": 2020, "sha256": "same-source"}]))
        pd.DataFrame({"split": ["development"]}).to_csv(directory / "metrics.csv", index=False)
        pd.DataFrame(rows).to_csv(directory / "predictions.csv", index=False)
    return directories


def test_comparison_counts_games_and_checks_alignment(tmp_path):
    baseline, enhanced = reports(tmp_path)
    result = compare(baseline, enhanced, tmp_path / "comparison.md")
    assert result.games.eq(1).all()
    assert result.winner_accuracy.eq(1).all()
    assert result.team_points_mae.eq(1).all()
    changed = pd.read_csv(enhanced / "predictions.csv")
    changed.loc[0, "points"] = 99
    changed.to_csv(enhanced / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="identical games"):
        compare(baseline, enhanced, tmp_path / "invalid.md")


def test_comparison_rejects_different_sources(tmp_path):
    baseline, enhanced = reports(tmp_path)
    (enhanced / "data_manifest.json").write_text(json.dumps([{"season": 2020, "sha256": "other-source"}]))
    with pytest.raises(ValueError, match="identical historical source"):
        compare(baseline, enhanced, tmp_path / "invalid.md")


def test_comparison_rejects_holdout_tuning(tmp_path):
    baseline, enhanced = reports(tmp_path)
    pd.DataFrame({"split": ["holdout"]}).to_csv(enhanced / "metrics.csv", index=False)
    with pytest.raises(ValueError, match="development-only"):
        compare(baseline, enhanced, tmp_path / "invalid.md")
