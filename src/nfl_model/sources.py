"""Verified HTTPS adapters. Credentials and authenticated request URLs are never persisted."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pandas as pd
import requests
from nfl_model.betting import QUOTE_COLUMNS, validate_quotes

ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
TEAM_NAMES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "LA Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO",
    "New York Giants": "NYG", "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
ESPN_ALIASES = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX"}


class SourceUnavailable(RuntimeError):
    pass


def get_json(url: str, *, params=None):
    try:
        response = requests.get(url, params=params, timeout=(15, 45))
    except requests.RequestException:
        raise SourceUnavailable("Network connection unavailable; check allowed domains") from None
    if response.status_code in {401, 403}:
        raise SourceUnavailable(f"Access denied (HTTP {response.status_code}); check network policy, credentials, and subscription")
    if response.status_code == 429:
        raise SourceUnavailable("Provider rate limit reached; retry later")
    if not response.ok:
        raise SourceUnavailable(f"Provider returned HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError:
        raise SourceUnavailable("Provider returned invalid JSON") from None


def save_snapshot(directory: Path, name: str, payload):
    directory.mkdir(parents=True, exist_ok=True)
    # Unique names keep previous snapshots intact, including within the same minute.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{stamp}_{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def normalize_odds(payload, game: dict, observed_at: str) -> pd.DataFrame:
    rows = []
    events = payload.get("data", []) if isinstance(payload, dict) else payload
    for event in events:
        home = TEAM_NAMES.get(event.get("home_team"))
        away = TEAM_NAMES.get(event.get("away_team"))
        if home != game["home_team"] or away != game["away_team"]:
            continue
        if game.get("gameday") and pd.Timestamp(event["commence_time"]).tz_convert("America/New_York").date() != pd.Timestamp(game["gameday"]).date():
            continue
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                key = {"h2h": "moneyline", "spreads": "spread", "totals": "total"}.get(market["key"])
                if key is None:
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome["name"]
                    side = ("home" if name == event["home_team"] else "away" if name == event["away_team"]
                            else name.lower() if name in {"Over", "Under"} else None)
                    if side is None:
                        continue
                    updated = market.get("last_update") or bookmaker.get("last_update")
                    if not updated:
                        continue
                    rows.append({"game_id": game["game_id"], "bookmaker": bookmaker["key"],
                                 "market": key, "side": side, "point": outcome.get("point", 0),
                                 "american_odds": outcome["price"], "observed_at": observed_at,
                                 "commence_time": event["commence_time"], "source": "the-odds-api",
                                 "source_updated_at": updated})
    return validate_quotes(pd.DataFrame(rows, columns=QUOTE_COLUMNS))


def fetch_odds(game: dict, snapshots: Path, *, historical_at: str | None = None) -> pd.DataFrame:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise SourceUnavailable("ODDS_API_KEY is not configured; add it securely in environment settings")
    parameters = {"apiKey": key, "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american"}
    prefix = "historical/" if historical_at else ""
    if historical_at:
        timestamp = pd.Timestamp(historical_at)
        if timestamp.tzinfo is None:
            raise ValueError("Historical odds date requires a timezone")
        parameters["date"] = timestamp.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = get_json(f"https://api.the-odds-api.com/v4/{prefix}sports/americanfootball_nfl/odds", params=parameters)
    observed = payload.get("timestamp") if historical_at else datetime.now(timezone.utc).isoformat()
    if not observed:
        raise SourceUnavailable("Historical response did not contain a snapshot timestamp")
    if historical_at and pd.Timestamp(observed) > pd.Timestamp(historical_at):
        raise SourceUnavailable("Provider returned a historical snapshot later than the requested cutoff")
    quotes = normalize_odds(payload, game, observed)
    save_snapshot(snapshots, "historical_odds" if historical_at else "live_odds", payload)
    if quotes.empty:
        raise SourceUnavailable("Provider has no matching prices for this game")
    return quotes


def fetch_lineups(game: dict, snapshots: Path) -> dict:
    observed = datetime.now(timezone.utc).isoformat()
    scoreboard = get_json(f"{ESPN}/scoreboard", params={"dates": str(game["gameday"]).replace("-", "")})
    matched = None
    for event in scoreboard.get("events", []):
        for competition in event.get("competitions", []):
            teams = {ESPN_ALIASES.get(c["team"]["abbreviation"], c["team"]["abbreviation"]): c
                     for c in competition.get("competitors", [])}
            if set(teams) == {game["home_team"], game["away_team"]}:
                matched = (event, competition, teams)
    if matched is None:
        raise SourceUnavailable("ESPN has no matching scheduled game")
    event, competition, teams = matched
    context = {"source": "espn", "observed_at": observed, "commence_time": event["date"],
               "game_state": event.get("status", {}).get("type", {}).get("state"),
               "lineups_reviewed": False, "teams": {}, "venue": competition.get("venue", {})}
    for abbreviation, competitor in teams.items():
        team_id = competitor["team"]["id"]
        entry = {"injuries": [], "expected_qb": None, "depth_chart_is_confirmation": False}
        try:
            depth = get_json(f"{ESPN}/teams/{team_id}/depthcharts")
            roster = get_json(f"{ESPN}/teams/{team_id}/roster")
            for chart in depth.get("depthcharts", depth.get("depthchart", [])):
                for position in chart.get("positions", {}).values():
                    if position.get("position", {}).get("abbreviation", "").upper() != "QB":
                        continue
                    athletes = position.get("athletes", [])
                    if athletes:
                        athlete = athletes[0].get("athlete", athletes[0])
                        entry["expected_qb"] = {"espn_id": athlete.get("id"), "name": athlete.get("displayName"), "basis": "depth chart"}
                        break
                if entry["expected_qb"] is not None:
                    break
            for group in roster.get("athletes", []):
                for athlete in group.get("items", []):
                    for injury in athlete.get("injuries", []):
                        entry["injuries"].append({"player": athlete.get("displayName"), "espn_id": athlete.get("id"),
                                                 "status": injury.get("status"), "date": injury.get("date"),
                                                 "details": injury.get("details", {})})
            save_snapshot(snapshots, f"roster_depth_{abbreviation}", {"roster": roster, "depthchart": depth})
        except SourceUnavailable as error:
            entry["error"] = str(error)
        context["teams"][abbreviation] = entry
    save_snapshot(snapshots, "lineups", context)
    return context


def fetch_weather(game: dict, context: dict, snapshots: Path) -> dict:
    location = context.get("venue", {}).get("location", {})
    if location.get("latitude") is not None and location.get("longitude") is not None:
        latitude, longitude = location["latitude"], location["longitude"]
    elif game["home_team"] == "LA":
        latitude, longitude = 33.9535, -118.3392
    else:
        raise SourceUnavailable("Stadium coordinates are unavailable; supply a venue location in context")
    kickoff = context.get("commence_time")
    if not kickoff:
        raise SourceUnavailable("A verified kickoff timestamp is required for weather")
    payload = get_json("https://api.open-meteo.com/v1/forecast", params={
        "latitude": latitude, "longitude": longitude, "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "UTC"})
    hourly = payload.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []), utc=True)
    if len(times) == 0:
        raise SourceUnavailable("Weather provider returned no forecast hours")
    difference = abs(times - pd.Timestamp(kickoff))
    index = int(difference.argmin())
    if difference[index] > pd.Timedelta(hours=1):
        raise SourceUnavailable("Kickoff is outside the weather forecast range")
    result = {"source": "open-meteo", "observed_at": datetime.now(timezone.utc).isoformat(),
              "forecast_hour": times[index].isoformat(), "temperature_f": hourly["temperature_2m"][index],
              "wind_mph": hourly["wind_speed_10m"][index], "precipitation_mm": hourly["precipitation"][index],
              "venue_indoor": context.get("venue", {}).get("indoor"), "used_in_score_model": False}
    save_snapshot(snapshots, "weather", {"summary": result, "raw": payload})
    return result


def map_quarterback_ids(context: dict, roster: pd.DataFrame) -> dict:
    """Map provider IDs to nflverse IDs; never infer a starter from an active roster."""
    from copy import deepcopy
    context = deepcopy(context)
    for team, entry in context.get("teams", {}).items():
        quarterback = entry.get("expected_qb") or {}
        identifier = quarterback.get("espn_id")
        if identifier is None:
            continue
        matches = roster.loc[roster.position.eq("QB") & roster.team.eq(team)
                             & roster.espn_id.astype(str).str.replace(r"\.0$", "", regex=True).eq(str(identifier))]
        if len(matches) == 1 and pd.notna(matches.iloc[0].gsis_id):
            entry["expected_qb_gsis_id"] = matches.iloc[0].gsis_id
    return context


def load_roster(season: int, directory: Path, *, refresh=False) -> pd.DataFrame:
    import hashlib
    import uuid
    import pyarrow.parquet as pq
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"roster_{season}.parquet"
    if target.exists() and not refresh:
        return pd.read_parquet(target)
    temporary = target.with_name(f"roster_{season}.{uuid.uuid4().hex}.part")
    try:
        try:
            response = requests.get(f"https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.parquet", timeout=(15, 60))
        except requests.RequestException:
            raise SourceUnavailable("Roster download unavailable") from None
        if not response.ok:
            raise SourceUnavailable(f"Roster provider returned HTTP {response.status_code}")
        temporary.write_bytes(response.content)
        try:
            pq.read_metadata(temporary)
            roster = pd.read_parquet(temporary)
            required = {"position", "team", "espn_id", "gsis_id"}
            if not required.issubset(roster.columns):
                raise ValueError("Missing roster fields")
        except (ValueError, OSError):
            raise SourceUnavailable("Roster file is invalid; previous cache retained") from None
        if target.exists():
            previous = target.read_bytes()
            archive = directory / "roster_archive"
            archive.mkdir(exist_ok=True)
            archived = archive / f"roster_{season}_{hashlib.sha256(previous).hexdigest()[:16]}.parquet"
            if not archived.exists():
                archived.write_bytes(previous)
        temporary.replace(target)
        return roster
    finally:
        temporary.unlink(missing_ok=True)
