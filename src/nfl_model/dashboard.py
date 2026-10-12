"""Streamlit interface for research forecasts, live prices, diagnostics, and paper bets."""
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from nfl_model.betting import price_quotes, validate_quotes
from nfl_model.paper import PaperLedger
from nfl_model.workflow import ROOT, update

st.set_page_config(page_title="NFL Forecast Lab", page_icon="🏈", layout="wide")
st.title("NFL Forecast Lab")
st.caption("Ridge and Poisson forecasts • Research estimates • Paper betting only")

schedule_files = list((ROOT / "data/schedules").glob("*.csv"))
schedule = pd.read_csv(max(schedule_files, key=lambda p: p.stat().st_mtime)) if schedule_files else pd.DataFrame()
chosen_date = st.sidebar.date_input("Game date", datetime.now(ZoneInfo("America/Chicago")).date())
selected = schedule.loc[schedule.gameday.eq(str(chosen_date))] if not schedule.empty else pd.DataFrame()
if selected.empty:
    st.info("No scheduled games found for this date. Run the data pipeline or choose another date.")
    st.stop()
labels = {f"{row.away_team} at {row.home_team}": row.game_id for row in selected.itertuples()}
label = st.sidebar.selectbox("Matchup", list(labels))
game = selected.loc[selected.game_id.eq(labels[label])].iloc[0].to_dict()
quote_upload = st.sidebar.file_uploader("Timestamped odds CSV", type=["csv"])
context_upload = st.sidebar.file_uploader("Reviewed lineup context JSON", type=["json"])
st.sidebar.caption("Set API credentials securely in environment settings. Do not upload credentials here.")
if st.sidebar.button("Refresh forecast and sources", type="primary"):
    imports = ROOT / "data/imports"
    imports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    quotes_path, context_path = None, None
    try:
        if quote_upload:
            quotes = validate_quotes(pd.read_csv(quote_upload))
            quotes_path = imports / f"{stamp}_quotes.csv"
            quotes.to_csv(quotes_path, index=False)
        if context_upload:
            supplied = json.load(context_upload)
            # Only explicit context fields enter the forecast; credentials are never accepted.
            fields = {"source", "observed_at", "commence_time", "lineups_reviewed", "teams", "weather", "venue"}
            if set(supplied) - fields:
                raise ValueError("Context file contains unsupported fields")
            context_path = imports / f"{stamp}_context.json"
            context_path.write_text(json.dumps(supplied))
        with st.spinner("Updating data and generating the forecast…"):
            update(str(chosen_date), game["away_team"], game["home_team"], refresh=True,
                   quotes_file=quotes_path, context_file=context_path)
        st.rerun()
    except (ValueError, RuntimeError) as error:
        st.error(str(error))

state_file = ROOT / "artifacts/dashboard/state.json"
state = json.loads(state_file.read_text()) if state_file.exists() else {}
if state.get("game_id") == game["game_id"]:
    run_dir = ROOT / state["run_dir"]
else:
    run_dir = ROOT / "artifacts/upcoming" / game["game_id"]
forecast_file = run_dir / "prediction.json"
if not forecast_file.exists():
    st.info("Choose Refresh forecast and sources to generate this matchup.")
    st.stop()
forecast = json.loads(forecast_file.read_text())
if forecast.get("game_id") != game["game_id"]:
    st.error("Forecast matchup does not match the selected game.")
    st.stop()
context = forecast.get("context", {})
starters_available = all(context.get("teams", {}).get(team, {}).get("expected_qb_gsis_id")
                         for team in [game["home_team"], game["away_team"]])
context_digest = hashlib.sha256(json.dumps({"teams": context.get("teams", {}), "weather": context.get("weather", {})},
                                          sort_keys=True).encode()).hexdigest()[:12]
reviewed = st.sidebar.checkbox("I reviewed the listed starters, injuries, and weather.",
                                value=bool(context.get("lineups_reviewed", False)), disabled=not starters_available,
                                key=f"review_{game['game_id']}_{context_digest}")
if reviewed and starters_available:
    forecast["context"]["lineups_reviewed"] = True
    forecast["context"]["reviewed_at"] = datetime.now(timezone.utc).isoformat()

closing_file = ROOT / "artifacts/closing-reference/summary.json"
if closing_file.exists():
    closing = json.loads(closing_file.read_text())
    if closing.get("roi") is not None:
        st.warning(f"Historical closing-line diagnostic return: {closing['roi']:.1%}. It has no quote timestamps and does not establish an actionable or profitable strategy.")

forecast_tab, odds_tab, probability_tab, paper_tab, sources_tab = st.tabs(
    ["Forecast", "Odds and value", "Probability validation", "Paper bets", "Data status"])
calibration_file = ROOT / "artifacts/probability/calibration.joblib"
quotes_file = run_dir / "quotes.csv"
quotes = pd.read_csv(quotes_file) if quotes_file.exists() and quotes_file.stat().st_size > 1 else pd.DataFrame()
priced = pd.DataFrame()
if not quotes.empty and calibration_file.exists():
    try:
        priced = price_quotes(forecast, quotes, joblib.load(calibration_file))
    except ValueError as error:
        st.error(str(error))

with forecast_tab:
    st.subheader(label)
    models = pd.DataFrame(forecast["models"])
    cols = st.columns(3)
    cols[0].metric(f"{game['away_team']} expected points", f"{models.away_points.mean():.1f}")
    cols[1].metric(f"{game['home_team']} expected points", f"{models.home_points.mean():.1f}")
    cols[2].metric("Expected total", f"{models.total_points.mean():.1f}")
    st.dataframe(models.rename(columns={"model": "Model", "away_points": f"{game['away_team']} points",
                                        "home_points": f"{game['home_team']} points", "away_margin": f"{game['away_team']} margin",
                                        "total_points": "Total"}), hide_index=True)
    probabilities = forecast.get("probabilities")
    if probabilities:
        st.write(f"Research win estimates: {game['home_team']} {probabilities['home_win']:.1%}; "
                 f"{game['away_team']} {probabilities['away_win']:.1%}; tie {probabilities['tie']:.1%}.")
    st.caption(f"Generated {pd.Timestamp(forecast['generated_at_utc']).tz_convert('America/Chicago').strftime('%b %d, %Y %I:%M %p %Z')}. "
               f"Statistical history through {forecast['latest_observed_game_date']}; scoring models trained through {forecast['trained_through_season']}.")
    context = forecast.get("context", {})
    st.write("Lineups reviewed" if context.get("lineups_reviewed") else "Tonight's lineups have not been reviewed.")
    for team, information in context.get("teams", {}).items():
        quarterback = information.get("expected_qb") or {}
        st.write(f"{team}: expected quarterback {quarterback.get('name') or 'not confirmed'}.")
        if information.get("injuries"):
            st.dataframe(pd.DataFrame(information["injuries"]), hide_index=True)
    if context.get("weather"):
        weather = context["weather"]
        st.write(f"Kickoff forecast: {weather.get('temperature_f', '—')} °F, wind {weather.get('wind_mph', '—')} mph, "
                 f"precipitation {weather.get('precipitation_mm', '—')} mm.")
    st.caption("Starter overrides use prior player statistics. Other injuries and weather are displayed context, not fitted point adjustments.")

with odds_tab:
    st.subheader("Available prices and research value")
    if priced.empty:
        st.info("No verified, timestamped sportsbook prices are available. Configure the odds API or import a timestamped CSV.")
        if pd.notna(game.get("spread_line")) and pd.notna(game.get("total_line")):
            st.caption(f"Untimestamped schedule reference only: home handicap {-game['spread_line']:+g}; total {game['total_line']:g}. These are not ranked as live bets.")
    else:
        display = priced[["bookmaker", "market", "side", "point", "american_odds", "p_win", "p_push", "estimated_ev", "status"]].copy()
        display["side"] = display.side.replace({"home": game["home_team"], "away": game["away_team"]})
        st.dataframe(display, hide_index=True, column_config={
            "p_win": st.column_config.NumberColumn("Win probability", format="percent"),
            "p_push": st.column_config.NumberColumn("Push probability", format="percent"),
            "estimated_ev": st.column_config.NumberColumn("Estimated return / unit", format="percent")})
        eligible = priced.loc[priced.research_candidate]
        st.write(f"{len(eligible)} eligible research paper candidates." if len(eligible) else "No qualifying paper bets.")
        st.caption("Value estimates include the offered price and pushes. No-vig comparisons require opposing prices from the same book, line, and snapshot.")

with probability_tab:
    st.subheader("Chronological probability validation")
    metrics_file = ROOT / "artifacts/probability/metrics.csv"
    if metrics_file.exists():
        metrics = pd.read_csv(metrics_file)
        st.dataframe(metrics, hide_index=True)
        reliability = pd.read_csv(ROOT / "artifacts/probability/reliability.csv")
        market = st.selectbox("Probability market", ["moneyline", "spread", "total"])
        chosen = reliability.loc[reliability.market.eq(market)]
        st.scatter_chart(chosen, x="predicted", y="actual", color="model", size="games")
        st.caption("Lower Brier/log loss is better. Pushes are excluded from these metrics. Spread/total calibration uses untimestamped reference closing lines. All displayed seasons are development evidence.")
    else:
        st.info("Run nfl-calibrate to generate probability validation.")

with paper_tab:
    ledger = PaperLedger(ROOT / "data/paper.sqlite")
    summary = ledger.summary()
    cols = st.columns(3)
    cols[0].metric("Paper bets", summary["recorded"])
    cols[1].metric("Pending", summary["pending"])
    cols[2].metric("Net units", f"{summary['profit_units']:.2f}")
    st.caption("One flat-unit paper bet per game. Existing entries retain their original price and forecast. No real wagers are placed.")
    eligible = priced.loc[priced.research_candidate] if not priced.empty else pd.DataFrame()
    if st.button("Log top paper candidate", disabled=eligible.empty):
        try:
            recorded = ledger.record_bet(eligible.iloc[0].to_dict(), forecast)
            st.success("Paper bet recorded." if recorded else "A paper bet already exists for this game.")
            st.rerun()
        except ValueError as error:
            st.error(str(error))
    bets = ledger.bets()
    if bets.empty:
        st.info("No prospective paper bets recorded yet. Results appear after eligible bets are logged and games finish.")
    else:
        st.dataframe(bets.drop(columns=["forecast_json", "forecast_sha256"]), hide_index=True)
        settled = bets.loc[bets.result.notna()].copy()
        if not settled.empty:
            settled["net_units"] = settled.profit_units.cumsum()
            st.line_chart(settled.set_index("created_at")[["net_units"]])

with sources_tab:
    st.subheader("Source availability")
    for name, status in forecast.get("source_status", {}).items():
        st.write(f"**{name.replace('_', ' ').title()}**: {status}")
    st.write("Stale prices, started games, missing lineup review, small estimated edges, and disagreement between models prevent paper candidates.")
    st.caption("Enable ESPN, Open-Meteo, and The Odds API access in environment settings. Store ODDS_API_KEY securely there.")


@st.fragment(run_every="60s")
def refresh_display():
    """Refresh the view and stale-price gates without spending provider API credits."""
    current = datetime.now(timezone.utc).timestamp()
    previous = st.session_state.get("last_display_refresh", current)
    st.session_state.setdefault("last_display_refresh", current)
    if current - previous >= 55:
        st.session_state["last_display_refresh"] = current
        st.rerun()


refresh_display()
