"""Download public nflverse data over verified HTTPS and aggregate team games."""
from pathlib import Path
import pandas as pd
import requests

BASE = "https://github.com/nflverse/nflverse-data/releases/download/pbp"
COLUMNS = [
    "game_id", "season", "week", "game_date", "season_type", "home_team", "away_team",
    "home_score", "away_score", "posteam", "defteam", "epa", "pass_attempt",
    "rush_attempt", "sack", "yards_gained", "qb_kneel", "qb_spike", "down",
    "passer_player_id",
]
METRICS = ["epa", "pass_epa", "rush_epa", "success", "explosive", "sack_rate", "early_epa", "plays"]
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX"}


def download_season(season: int, cache: Path, *, refresh: bool = False) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"play_by_play_{season}.parquet"
    if target.exists() and not refresh:
        # Detect truncated or invalid cache entries before reusing them.
        import pyarrow.parquet as pq
        pq.read_metadata(target)
        return target
    import uuid
    temporary = target.with_name(f"{target.stem}.{uuid.uuid4().hex}.part")
    try:
        with requests.get(f"{BASE}/{target.name}", stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        import pyarrow.parquet as pq
        pq.read_metadata(temporary)
        if target.exists():
            import hashlib
            archive = cache / "archive"
            archive.mkdir(exist_ok=True)
            previous = target.read_bytes()
            digest = hashlib.sha256(previous).hexdigest()[:16]
            preserved = archive / f"{target.stem}_{digest}.parquet"
            if not preserved.exists():
                preserved.write_bytes(previous)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def aggregate_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per team/game. Outcomes are separate from shifted pregame features."""
    pbp = pbp.loc[pbp.season_type.eq("REG")].copy()
    for column in ["home_team", "away_team", "posteam", "defteam"]:
        pbp[column] = pbp[column].replace(TEAM_ALIASES)
    game_columns = ["game_id", "season", "week", "game_date", "home_team", "away_team", "home_score", "away_score"]
    games = pbp[game_columns].drop_duplicates("game_id").copy()
    games["game_date"] = pd.to_datetime(games.game_date)
    games = games.dropna(subset=["home_score", "away_score"])
    valid = ((pbp.pass_attempt.eq(1) | pbp.rush_attempt.eq(1))
             & ~pbp.qb_kneel.eq(1) & ~pbp.qb_spike.eq(1)
             & pbp.epa.notna() & pbp.posteam.notna() & pbp.defteam.notna())
    plays = pbp.loc[valid].copy()
    plays["pass_epa"] = plays.epa.where(plays.pass_attempt.eq(1))
    plays["rush_epa"] = plays.epa.where(plays.rush_attempt.eq(1))
    plays["success"] = plays.epa.gt(0).astype(float)
    plays["explosive"] = ((plays.pass_attempt.eq(1) & plays.yards_gained.ge(20))
                          | (plays.rush_attempt.eq(1) & plays.yards_gained.ge(10))).astype(float)
    plays["sack_rate"] = plays.sack.where(plays.pass_attempt.eq(1))
    plays["early_epa"] = plays.epa.where(plays.down.le(2))
    frames = []
    for side, opponent in [("home_team", "away_team"), ("away_team", "home_team")]:
        frame = games.rename(columns={side: "team", opponent: "opponent"}).copy()
        frame["is_home"] = int(side == "home_team")
        frame["points"] = frame.home_score if side == "home_team" else frame.away_score
        frame["points_allowed"] = frame.away_score if side == "home_team" else frame.home_score
        frames.append(frame)
    team_games = pd.concat(frames, ignore_index=True)
    for role, prefix in [("posteam", "off"), ("defteam", "def")]:
        grouped = plays.groupby(["game_id", role])
        means = grouped[METRICS[:-1]].mean()
        means["plays"] = grouped.size()
        means = means.add_prefix(prefix + "_").reset_index().rename(columns={role: "team"})
        team_games = team_games.merge(means, on=["game_id", "team"], how="left", validate="one_to_one")
    # Identify the primary passer retrospectively; features.py only uses this AFTER the game.
    if "passer_player_id" in plays:
        passes = plays.loc[plays.pass_attempt.eq(1) & plays.passer_player_id.notna()]
        quarterbacks = passes.groupby(["game_id", "posteam", "passer_player_id"]).agg(
            primary_qb_epa=("epa", "mean"), primary_qb_dropbacks=("epa", "size")).reset_index()
        quarterbacks = quarterbacks.sort_values(
            ["primary_qb_dropbacks", "passer_player_id"], ascending=[False, True])
        quarterbacks = quarterbacks.drop_duplicates(["game_id", "posteam"]).rename(
            columns={"posteam": "team", "passer_player_id": "primary_qb_id"})
        team_games = team_games.merge(quarterbacks, on=["game_id", "team"], how="left", validate="one_to_one")
    return team_games.sort_values(["game_date", "game_id", "team"]).reset_index(drop=True)
