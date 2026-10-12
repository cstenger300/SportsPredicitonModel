"""Time-safe rolling features: a game never contributes to its own inputs."""
import pandas as pd
from nfl_model.data import METRICS
from nfl_model.ratings import opponent_adjusted_ratings, RATING_FEATURES
from nfl_model.quarterbacks import quarterback_history, QB_FEATURES

TEAM_METRICS = [f"{role}_{metric}" for role in ["off", "def"] for metric in METRICS]


def build_features(team_games: pd.DataFrame, feature_set: str = "baseline") -> tuple[pd.DataFrame, list[str]]:
    if feature_set not in {"baseline", "enhanced"}:
        raise ValueError("feature_set must be baseline or enhanced")
    frame = team_games.sort_values(["game_date", "game_id", "team"]).reset_index(drop=True).copy()
    feature_names = []
    groups = frame.groupby("team", sort=False)
    # Carry information across season boundaries; rest is capped to avoid encoding a long offseason.
    for metric in TEAM_METRICS:
        name = f"recent_{metric}"
        frame[name] = groups[metric].transform(lambda x: x.shift(1).ewm(alpha=0.25, adjust=False).mean())
        feature_names.append(name)
    frame["rest_days"] = groups.game_date.diff().dt.days.clip(upper=21)
    frame["prior_games"] = groups.cumcount()
    feature_names += ["rest_days", "prior_games"]
    if feature_set == "enhanced":
        frame[RATING_FEATURES] = opponent_adjusted_ratings(frame)
        frame[QB_FEATURES] = quarterback_history(frame)
        feature_names += RATING_FEATURES + QB_FEATURES
    opponent_features = frame[["game_id", "team"] + feature_names].rename(
        columns={"team": "opponent", **{name: f"opponent_{name}" for name in feature_names}})
    frame = frame.merge(opponent_features, on=["game_id", "opponent"], validate="one_to_one")
    features = feature_names + [f"opponent_{name}" for name in feature_names] + ["is_home"]
    return frame, features
