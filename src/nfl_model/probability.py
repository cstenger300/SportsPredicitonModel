"""Score distributions and separately fitted market probability calibrators."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression


@dataclass
class ScoreDistribution:
    home: np.ndarray
    away: np.ndarray
    mass: np.ndarray

    def probabilities(self, market: str, side: str, point: float = 0.0) -> dict:
        if market == "moneyline":
            value = self.home - self.away if side == "home" else self.away - self.home
        elif market == "spread":
            value = (self.home - self.away if side == "home" else self.away - self.home) + point
        elif market == "total":
            value = self.home + self.away - point
            if side == "under":
                value = -value
        else:
            raise ValueError("Unknown market")
        return {"win": float(self.mass[value > 1e-9].sum()),
                "push": float(self.mass[np.abs(value) <= 1e-9].sum()),
                "loss": float(self.mass[value < -1e-9].sum())}


def score_distribution(kind: str, home_mean: float, away_mean: float, residuals: pd.DataFrame) -> ScoreDistribution:
    if not np.isfinite([home_mean, away_mean]).all() or residuals.empty:
        raise ValueError("Finite means and out-of-sample residual history are required")
    if kind == "ridge":
        home = np.maximum(0, np.rint(home_mean + residuals.home_residual.to_numpy()))
        away = np.maximum(0, np.rint(away_mean + residuals.away_residual.to_numpy()))
        mass = np.ones(len(home), dtype=float) / len(home)
    elif kind == "poisson":
        home_mean, away_mean = max(home_mean, 1e-6), max(away_mean, 1e-6)
        maximum = int(max(100, np.ceil(max(home_mean, away_mean) + 12 * np.sqrt(max(home_mean, away_mean)))))
        h, a = np.meshgrid(np.arange(maximum + 1), np.arange(maximum + 1), indexing="ij")
        home, away = h.ravel(), a.ravel()
        mass = (poisson.pmf(h, home_mean) * poisson.pmf(a, away_mean)).ravel()
        mass /= mass.sum()
    else:
        raise ValueError("Model must be ridge or poisson")
    # Final NFL ties are rare because of overtime; independent point counts overstate them.
    tied = home == away
    historical_tie_rate = float(residuals.actual_home.eq(residuals.actual_away).mean())
    if not tied.any() and historical_tie_rate > 0:
        tie_score = max(0, int(np.rint((home_mean + away_mean) / 2)))
        home = np.append(home, tie_score)
        away = np.append(away, tie_score)
        mass = np.append(mass * (1 - historical_tie_rate), historical_tie_rate)
        tied = home == away
    if tied.all() and historical_tie_rate < 1:
        raise ValueError("Residual distribution has no decisive score outcomes")
    if tied.any() and (~tied).any():
        mass[tied] *= historical_tie_rate / mass[tied].sum() if mass[tied].sum() else 0
        mass[~tied] *= (1 - historical_tie_rate) / mass[~tied].sum()
    mass /= mass.sum()
    return ScoreDistribution(home, away, mass)


def conditional_win(probabilities: dict) -> float:
    decisive = probabilities["win"] + probabilities["loss"]
    return probabilities["win"] / decisive if decisive else 0.5


def logit_input(probabilities):
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(values / (1 - values)).reshape(-1, 1)


def fit_calibrator(probabilities, outcomes):
    labels = np.asarray(outcomes, dtype=int)
    if len(labels) < 50 or len(np.unique(labels)) < 2:
        return None
    return LogisticRegression(C=1.0, max_iter=1000).fit(logit_input(probabilities), labels)


def apply_calibrator(probabilities: dict, calibrator, *, reverse: bool = False) -> dict:
    if calibrator is None:
        return {**probabilities, "calibrated": False}
    base = conditional_win(probabilities)
    conditional = float(calibrator.predict_proba(logit_input([1 - base if reverse else base]))[0, 1])
    if reverse:
        conditional = 1 - conditional
    push = probabilities["push"]
    return {"win": conditional * (1 - push), "push": push,
            "loss": (1 - conditional) * (1 - push), "calibrated": True}
