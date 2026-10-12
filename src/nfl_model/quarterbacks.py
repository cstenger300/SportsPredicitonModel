"""Prior-game quarterback proxies, never actual current-game starter information."""
import numpy as np
import pandas as pd

QB_FEATURES = ["previous_qb_recent_epa", "previous_qb_prior_dropbacks", "previous_qb_days_since_play"]


def quarterback_history(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"primary_qb_id", "primary_qb_epa", "primary_qb_dropbacks"}
    if not required.issubset(frame.columns):
        raise ValueError("Enhanced features require primary-quarterback aggregates from play-by-play")
    result = pd.DataFrame(np.nan, index=frame.index, columns=QB_FEATURES)
    previous_for_team = {}
    history = {}
    # Predict every game on a date before observing any game on that date.
    for date, games in frame.groupby("game_date", sort=True):
        for index, row in games.iterrows():
            qb = previous_for_team.get(row.team)
            if qb in history:
                epa, dropbacks, last_date = history[qb]
                result.loc[index, QB_FEATURES] = [epa, dropbacks, min((date - last_date).days, 365)]
        for row in games.itertuples():
            if pd.isna(row.primary_qb_id) or pd.isna(row.primary_qb_epa):
                continue
            qb = row.primary_qb_id
            old = history.get(qb)
            epa = float(row.primary_qb_epa) if old is None else 0.25 * row.primary_qb_epa + 0.75 * old[0]
            attempts = float(row.primary_qb_dropbacks) + (0.0 if old is None else old[1])
            history[qb] = (epa, attempts, date)
            previous_for_team[row.team] = qb
    return result
