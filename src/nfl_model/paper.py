"""Append-only paper bets and idempotent settlement. No real wagers are placed."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import pandas as pd
from nfl_model.betting import validate_quotes, settle, unit_profit


class PaperLedger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS quotes (
                    quote_id TEXT PRIMARY KEY, game_id TEXT NOT NULL, observed_at TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bets (
                    bet_id INTEGER PRIMARY KEY, game_id TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
                    bookmaker TEXT NOT NULL, market TEXT NOT NULL, side TEXT NOT NULL, point REAL NOT NULL,
                    american_odds REAL NOT NULL, observed_at TEXT NOT NULL, commence_time TEXT NOT NULL,
                    probability_win REAL NOT NULL, probability_push REAL NOT NULL, estimated_ev REAL NOT NULL,
                    forecast_sha256 TEXT NOT NULL, forecast_json TEXT NOT NULL,
                    result TEXT, profit_units REAL, settled_at TEXT
                );
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def record_quotes(self, quotes: pd.DataFrame):
        quotes = validate_quotes(quotes)
        with self.connect() as db:
            for record in quotes.to_dict("records"):
                payload = json.dumps(record, default=str, sort_keys=True)
                identifier = hashlib.sha256(payload.encode()).hexdigest()
                db.execute("INSERT OR IGNORE INTO quotes VALUES (?, ?, ?, ?)",
                           [identifier, record["game_id"], str(record["observed_at"]), payload])

    def record_bet(self, candidate: dict, forecast: dict, now=None) -> bool:
        now = pd.Timestamp(now or datetime.now(timezone.utc))
        if now.tzinfo is None or now >= pd.Timestamp(candidate["commence_time"]):
            raise ValueError("Paper bets must be recorded before kickoff with a timezone")
        if not bool(candidate.get("research_candidate", False)):
            raise ValueError("This quote is not an eligible paper candidate")
        if not forecast.get("context", {}).get("lineups_reviewed", False):
            raise ValueError("Lineup review is required before logging a paper bet")
        if candidate["game_id"] != forecast["game_id"]:
            raise ValueError("Forecast/quote matchup mismatch")
        validate_quotes(pd.DataFrame([candidate]))
        if (now - pd.Timestamp(candidate["observed_at"])).total_seconds() > 15 * 60 or (now - pd.Timestamp(candidate["source_updated_at"])).total_seconds() > 15 * 60:
            raise ValueError("Quote is stale; refresh before logging")
        if pd.Timestamp(candidate["observed_at"]) > now + pd.Timedelta(seconds=60):
            raise ValueError("Quote observation is in the future")
        payload = json.dumps(forecast, sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self.connect() as db:
            cursor = db.execute("""INSERT OR IGNORE INTO bets
                (game_id,created_at,bookmaker,market,side,point,american_odds,observed_at,commence_time,
                 probability_win,probability_push,estimated_ev,forecast_sha256,forecast_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [candidate["game_id"], now.isoformat(), candidate["bookmaker"], candidate["market"], candidate["side"],
                 float(candidate["point"]), float(candidate["american_odds"]), str(candidate["observed_at"]),
                 str(candidate["commence_time"]), float(candidate["p_win"]), float(candidate["p_push"]),
                 float(candidate["estimated_ev"]), digest, payload])
            return cursor.rowcount == 1

    def bets(self) -> pd.DataFrame:
        with self.connect() as db:
            return pd.read_sql_query("SELECT * FROM bets ORDER BY created_at, bet_id", db)

    def settle_from_schedule(self, schedule: pd.DataFrame, now=None) -> int:
        now = pd.Timestamp(now or datetime.now(timezone.utc))
        if now.tzinfo is None:
            raise ValueError("now must have a timezone")
        # Conservative finality: only scores on dates strictly before today, never an in-progress game.
        completed = schedule.loc[pd.to_datetime(schedule.gameday).dt.date.lt(now.tz_convert("America/New_York").date())
                                 & schedule.home_score.notna() & schedule.away_score.notna()].set_index("game_id")
        count = 0
        with self.connect() as db:
            for bet in self.bets().itertuples():
                if pd.notna(bet.result) or bet.game_id not in completed.index:
                    continue
                game = completed.loc[bet.game_id]
                result = settle(bet.market, bet.side, bet.point, game.home_score, game.away_score)
                db.execute("UPDATE bets SET result=?, profit_units=?, settled_at=? WHERE bet_id=? AND result IS NULL",
                           [result, unit_profit(result, bet.american_odds), now.isoformat(), bet.bet_id])
                count += 1
        return count

    def summary(self) -> dict:
        frame = self.bets()
        settled = frame.loc[frame.result.notna()].sort_values(["commence_time", "bet_id"])
        profits = settled.profit_units.astype(float)
        cumulative = profits.cumsum()
        drawdown = (cumulative.cummax().clip(lower=0) - cumulative).max() if len(cumulative) else 0.0
        return {"recorded": len(frame), "pending": int(frame.result.isna().sum()), "settled": len(settled),
                "profit_units": float(profits.sum()), "roi": float(profits.mean()) if len(profits) else None,
                "max_drawdown_units": float(drawdown)}
