"""Timestamp validation, sportsbook pricing, settlement, and conservative research candidates."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd

QUOTE_COLUMNS = ["game_id", "bookmaker", "market", "side", "point", "american_odds",
                 "observed_at", "commence_time", "source", "source_updated_at"]
SIDES = {"moneyline": {"home", "away"}, "spread": {"home", "away"}, "total": {"over", "under"}}


def decimal_odds(american: float) -> float:
    if not np.isfinite(american) or abs(american) < 100:
        raise ValueError("American odds must be finite and have magnitude at least 100")
    return 1 + american / 100 if american > 0 else 1 + 100 / abs(american)


def expected_value(probabilities: dict, american: float) -> float:
    return probabilities["win"] * (decimal_odds(american) - 1) - probabilities["loss"]


def settle(market: str, side: str, point: float, home_score: float, away_score: float) -> str:
    if market not in SIDES or side not in SIDES[market]:
        raise ValueError("Invalid market/side")
    if market == "moneyline":
        margin = home_score - away_score if side == "home" else away_score - home_score
    elif market == "spread":
        margin = (home_score - away_score if side == "home" else away_score - home_score) + point
    else:
        margin = home_score + away_score - point
        if side == "under":
            margin = -margin
    return "win" if margin > 1e-9 else "loss" if margin < -1e-9 else "push"


def unit_profit(result: str, american: float) -> float:
    return decimal_odds(american) - 1 if result == "win" else -1.0 if result == "loss" else 0.0


def validate_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    missing = set(QUOTE_COLUMNS) - set(quotes.columns)
    if missing:
        raise ValueError("Missing quote columns: " + ", ".join(sorted(missing)))
    frame = quotes[QUOTE_COLUMNS].copy()
    if frame.empty:
        return frame
    for name in ["game_id", "bookmaker", "source"]:
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {name}")
    for name in ["observed_at", "commence_time", "source_updated_at"]:
        # Naive timestamps do not establish a reliable quote cutoff.
        if not frame[name].astype(str).str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True).all():
            raise ValueError(f"{name} must include an explicit timezone")
        frame[name] = pd.to_datetime(frame[name], utc=True, errors="raise")
    if (frame.observed_at >= frame.commence_time).any():
        raise ValueError("Quotes must be observed before kickoff")
    if (frame.source_updated_at > frame.observed_at + pd.Timedelta(seconds=60)).any():
        raise ValueError("Source update cannot be after observation")
    for row in frame.itertuples():
        if row.market not in SIDES or row.side not in SIDES[row.market]:
            raise ValueError("Invalid quote market/side")
        decimal_odds(float(row.american_odds))
        if row.market != "moneyline" and not np.isfinite(float(row.point)):
            raise ValueError("Spread/total quotes require a finite point")
    frame["point"] = pd.to_numeric(frame.point).fillna(0.0)
    frame["american_odds"] = pd.to_numeric(frame.american_odds)
    return frame


def price_quotes(forecast: dict, quotes: pd.DataFrame, bundle: dict, now=None, *, max_age_minutes=15) -> pd.DataFrame:
    from nfl_model.probability import score_distribution, apply_calibrator
    quotes = validate_quotes(quotes)
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        raise ValueError("now must have a timezone")
    estimates = {m["model"]: m for m in forecast["models"]}
    distributions = {kind: score_distribution(kind, model["home_points"], model["away_points"], bundle["residuals"][kind])
                     for kind, model in estimates.items()}
    rows = []
    for quote in quotes.loc[quotes.game_id.eq(forecast["game_id"])].itertuples():
        probabilities = []
        model_evs = []
        for kind, distribution in distributions.items():
            raw = distribution.probabilities(quote.market, quote.side, float(quote.point))
            calibrated = apply_calibrator(raw, bundle["calibrators"].get((kind, quote.market)),
                                          reverse=quote.side in {"away", "under"})
            probabilities.append(calibrated)
            model_evs.append(expected_value(calibrated, quote.american_odds))
        average = {name: float(np.mean([p[name] for p in probabilities])) for name in ["win", "push", "loss"]}
        reasons = []
        if now >= quote.commence_time:
            reasons.append("game started")
        if quote.observed_at > now + pd.Timedelta(seconds=60):
            reasons.append("future quote timestamp")
        if (now - quote.observed_at).total_seconds() > max_age_minutes * 60 or (now - quote.source_updated_at).total_seconds() > max_age_minutes * 60:
            reasons.append("stale quote")
        generated = pd.Timestamp(forecast["generated_at_utc"])
        if generated > now + pd.Timedelta(seconds=60) or generated >= quote.commence_time or (now - generated).total_seconds() > 4 * 3600:
            reasons.append("stale or invalid forecast timestamp")
        if pd.Timestamp(forecast["feature_history_before"]).date() != quote.commence_time.tz_convert("America/New_York").date():
            reasons.append("forecast/game date mismatch")
        context = forecast.get("context", {})
        if not context.get("lineups_reviewed", False):
            reasons.append("lineups not reviewed")
        context_time = pd.to_datetime(context.get("observed_at"), utc=True, errors="coerce")
        if pd.isna(context_time) or context_time > now + pd.Timedelta(seconds=60) or (now - context_time).total_seconds() > 6 * 3600:
            reasons.append("lineup context missing or stale")
        if not all(context.get("teams", {}).get(team, {}).get("expected_qb_gsis_id")
                   for team in [forecast.get("home_team"), forecast.get("away_team")]):
            reasons.append("expected starters unavailable")
        if not all(p["calibrated"] for p in probabilities):
            reasons.append("probability calibration unavailable")
        if int(bundle["calibrated_through_season"]) >= int(forecast["game_id"].split("_")[0]):
            reasons.append("calibration includes prediction season")
        ev = expected_value(average, quote.american_odds)
        if ev < 0.03 or min(model_evs) <= 0:
            reasons.append("edge too small or models disagree")
        if context.get("unmodeled_qb_change", False):
            reasons.append("quarterback change not modeled")
        rows.append({**quote._asdict(), **{f"p_{key}": value for key, value in average.items()},
                     "implied_probability": 1 / decimal_odds(quote.american_odds), "estimated_ev": ev,
                     "research_candidate": not reasons, "status": "paper candidate" if not reasons else "; ".join(reasons),
                     "model_disagreement": float(max(model_evs) - min(model_evs))})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Remove vig only from paired, contemporaneous opposing outcomes at the same book and line.
    result["no_vig_probability"] = np.nan
    for _, group in result.groupby(["game_id", "bookmaker", "market", "observed_at"]):
        if len(group) != 2 or set(group.side) != SIDES[group.market.iloc[0]]:
            continue
        market = group.market.iloc[0]
        if market == "spread" and not np.isclose(group.point.sum(), 0):
            continue
        if market == "total" and not np.isclose(group.point.iloc[0], group.point.iloc[1]):
            continue
        result.loc[group.index, "no_vig_probability"] = group.implied_probability / group.implied_probability.sum()
    return result.sort_values("estimated_ev", ascending=False).reset_index(drop=True)
