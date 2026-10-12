"""Opponent-adjusted ratings fitted before each week's first game."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

RATING_TARGETS = ["epa", "pass_epa", "rush_epa"]
RATING_FEATURES = [f"adjusted_{role}_{metric}" for metric in RATING_TARGETS for role in ["off", "def_allowed"]]


def opponent_adjusted_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(np.nan, index=frame.index, columns=RATING_FEATURES)
    for _, week in frame.groupby(["season", "week"], sort=True):
        cutoff = week.game_date.min()
        history = frame.loc[(frame.game_date < cutoff) &
                            (frame.game_date >= cutoff - pd.Timedelta(days=730))].copy()
        targets = [f"off_{metric}" for metric in RATING_TARGETS]
        history = history.dropna(subset=targets + ["off_plays"])
        if history.empty:
            continue
        records = [{f"off:{row.team}": 1.0, f"def:{row.opponent}": 1.0, "home": float(row.is_home)}
                   for row in history.itertuples()]
        encoder = DictVectorizer(sparse=False)
        x = encoder.fit_transform(records)
        # Play counts represent sample size; recency has a fixed 365-day half-life.
        weights = history.off_plays.to_numpy() / 60 * np.exp2(
            -(cutoff - history.game_date).dt.days.to_numpy() / 365)
        model = Ridge(alpha=10.0).fit(x, history[targets], sample_weight=weights)
        positions = encoder.vocabulary_
        for index, row in week.iterrows():
            for target_index, metric in enumerate(RATING_TARGETS):
                for role, key in [("off", f"off:{row.team}"), ("def_allowed", f"def:{row.team}")]:
                    position = positions.get(key)
                    result.loc[index, f"adjusted_{role}_{metric}"] = (
                        model.coef_[target_index, position] if position is not None else 0.0)
    return result
