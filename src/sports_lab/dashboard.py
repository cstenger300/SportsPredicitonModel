"""Multi-sport research dashboard."""
from pathlib import Path
import pandas as pd
import streamlit as st
from sports_lab.core import METRICS, Ledger, fit_model, rank_quotes
from sports_lab import provider
from sports_lab.performance import ForecastStore
st.set_page_config(page_title='Sports Edge Lab', page_icon='📊', layout='wide')
st.title('Sports Edge Lab')
st.caption('NBA · NHL · NFL · NCAAB · Tennis | Advanced metrics · Sportsbook comparison · Paper results')
sport = st.sidebar.selectbox('Sport', list(METRICS))
st.sidebar.info('Moneylines only. Research probabilities are not calibrated confidence levels. No real wagers are placed.')
ledger = Ledger(Path(__file__).resolve().parents[2] / 'data/sports_lab/paper.sqlite')
forecast_store = ForecastStore(ledger.path)
board, tracker, performance_tab, help_tab = st.tabs(['Find value', 'Results tracker', 'Model performance', 'Data and methodology'])
key = provider.KEYS.get(sport)
with help_tab:
    st.subheader('Point-in-time advanced metrics')
    st.write(', '.join(METRICS[sport]))
    st.write('Each metric is the home minus away difference, using consistent units across history and upcoming games. Tennis uses provider home/away player assignments and surface-specific history. The model learns coefficient signs from data.')
    st.write('Include event_id, home, away, commence_time, metrics_as_of. History also requires home_win (0 or 1) and result_available_at. Dates must include timezones. Supply at least 100 completed events; exclude ties and voids. Use authentic archived pregame metrics: timestamp checks cannot detect fabricated or retrospectively revised inputs.')
    st.write('Standardized logistic regression trains on the earliest 80% and validates on the latest 20%, excluding results unavailable at the validation cutoff. Final forecasts refit on history available before upcoming metrics. Brier score is compared with a constant historical home-win baseline. This development check does not establish profitability or calibrated confidence.')
    st.write('Paper candidate: expected return at least 3%, price age at most 15 minutes, validation Brier better than baseline, and no metric beyond four training standard deviations. Rankings use expected return per unit; explanations show statistical log-odds contributions, not causes. Book prices must use the same full-game settlement rules. Review tennis retirement/abandonment rules manually.')
    for historical in [True, False]:
        cols = ['event_id','home','away','commence_time','metrics_as_of'] + METRICS[sport] + (['home_win','result_available_at'] if historical else [])
        label = 'Historical' if historical else 'Upcoming'
        st.download_button(label + ' metrics template', pd.DataFrame(columns=cols).to_csv(index=False), f'{sport}_{label}.csv')
    st.download_button('Results template', 'sport,event_id,status,winner,completed_at\n', 'results.csv')
    st.caption('Results: status=final, winner=home/away/void. Void ties/cancellations only according to the selected book’s rules. Scores coverage varies by sport and subscription; manual imports are supported.')
with board:
    st.subheader(f'{sport} · Predictions and prices')
    if sport == 'Tennis':
        if st.button('Load active tennis tournaments'):
            try:
                st.session_state['tournaments'] = provider.tournaments()
            except ValueError as e:
                st.error(str(e))
        keys = st.session_state.get('tournaments', [])
        key = st.selectbox('Tournament', keys) if keys else None
    if key and st.button('Fetch current sportsbook prices'):
        try:
            st.session_state[f'live_{sport}'] = provider.odds(key)
        except ValueError as e:
            st.error(str(e))
    live = st.session_state.get(f'live_{sport}')
    if live is not None:
        st.dataframe(live[0], hide_index=True)
        st.download_button('Export events for metric matching', live[0].to_csv(index=False), 'events.csv')
    history = st.file_uploader('Historical pregame metrics CSV', type='csv', key=f'h_{sport}')
    upcoming = st.file_uploader('Upcoming pregame metrics CSV', type='csv', key=f'u_{sport}')
    quotes_file = st.file_uploader('Optional odds CSV · overrides live prices', type='csv', key=f'q_{sport}')
    st.caption('Odds columns: event_id, bookmaker, side (home/away), american_odds, observed_at, source_updated_at. Use two-way full-game moneylines only.')
    if history and upcoming and st.button('Analyze matchups', type='primary'):
        try:
            forecasts, report = fit_model(pd.read_csv(history), pd.read_csv(upcoming), sport)
            quotes = pd.read_csv(quotes_file) if quotes_file else (live[1] if live is not None else pd.DataFrame())
            if live is not None and not quotes_file:
                for e in forecasts.to_dict('records'):
                    match = live[0].loc[live[0].event_id.eq(e['event_id'])]
                    if len(match)!=1 or any(match.iloc[0][c]!=e[c] for c in ['home','away']) or pd.Timestamp(match.iloc[0].commence_time)!=e['commence_time']:
                        raise ValueError('Upcoming IDs, participants and start times must match provider events.')
            ranked = rank_quotes(forecasts, quotes, report, sport) if not quotes.empty else pd.DataFrame()
            forecast_store.record(forecasts, report, sport)
            st.session_state[f'analysis_{sport}'] = (forecasts, report, ranked)
        except (ValueError, KeyError, TypeError) as e:
            st.error(str(e))
    analysis = st.session_state.get(f'analysis_{sport}')
    if analysis:
        forecasts, report, ranked = analysis
        c = st.columns(3)
        c[0].metric('Validation Brier · lower is better', f"{report['brier']:.3f}")
        c[1].metric('Baseline Brier', f"{report['baseline_brier']:.3f}")
        c[2].metric('Validation games', report['validation_events'])
        st.dataframe(forecasts[['home','away','home_probability','out_of_distribution','why']], hide_index=True)
        if ranked.empty:
            st.info('Fetch prices or import quotes to compare sportsbooks.')
        else:
            st.dataframe(ranked.drop(columns='validation'), hide_index=True)
            st.download_button('Export analysis', ranked.to_csv(index=False), 'analysis.csv')
            candidates = ranked.loc[ranked.eligible]
            if candidates.empty:
                st.info('No qualifying paper candidates in this snapshot.')
            else:
                selected = st.selectbox('Track a candidate', candidates.index, format_func=lambda i: f"{ranked.loc[i,'selection']} · {ranked.loc[i,'bookmaker']} · {ranked.loc[i,'american_odds']:+g}")
                if st.button('Record one-unit paper pick'):
                    try:
                        added = ledger.record(ranked.loc[selected].to_dict())
                        st.success('Recorded original price and forecast.' if added else 'A pick already exists for this event.')
                    except ValueError as e:
                        st.error(str(e))
with tracker:
    auto_scores = st.checkbox('Automatically refresh final scores every 15 minutes while open', value=False)
    st.caption('Uses The Odds API quota. Checks the selected sport/tournament; catches up on load when enabled. Older than three days or unsupported score feeds require reviewed CSV results.')
    @st.fragment(run_every='15m' if auto_scores else None)
    def refresh_scores_automatically():
        if auto_scores and key:
            try:
                results = provider.scores(sport, key)
                forecast_store.settle(results)
                count = ledger.settle(results)
                st.caption(f'Latest automatic check: {pd.Timestamp.now(tz="America/Chicago").strftime("%b %d %I:%M %p %Z")} · {count} newly settled picks. Reopen Results tracker to reload the table.')
            except ValueError as exc:
                st.warning(str(exc))
    refresh_scores_automatically()
    st.subheader('Paper results · original prices preserved')
    if key and st.button('Refresh final scores'):
        try:
            results = provider.scores(sport,key)
            forecast_store.settle(results)
            st.success(f'{ledger.settle(results)} picks settled.')
        except ValueError as e:
            st.error(str(e))
    uploaded = st.file_uploader('Reviewed final results CSV', type='csv')
    if uploaded and st.button('Settle imported results'):
        try:
            results = pd.read_csv(uploaded)
            forecast_store.settle(results)
            st.success(f'{ledger.settle(results)} picks settled.')
        except (ValueError, KeyError) as e:
            st.error(str(e))
    rows = ledger.rows()
    if rows.empty:
        st.info('Record a pregame paper candidate to start tracking.')
    else:
        rows = rows.loc[rows.sport.eq(sport)]
        settled = rows.loc[rows.result.isin(['won','lost'])]
        profit = float(rows.profit_units.sum())
        c = st.columns(4)
        c[0].metric('Won / lost', f"{sum(settled.result.eq('won'))} / {sum(settled.result.eq('lost'))}")
        c[1].metric('Pending', int(rows.result.isna().sum()))
        c[2].metric('Profit · units', f'{profit:+.2f}')
        c[3].metric('ROI · decisive stakes', f'{profit/len(settled):.1%}' if len(settled) else '—')
        st.dataframe(rows.drop(columns='validation',errors='ignore'), hide_index=True)
        st.download_button('Export paper history', rows.to_csv(index=False), 'paper_history.csv')

with performance_tab:
    st.subheader('Did the predictions improve?')
    st.write('Every analyzed game is saved before start, including games with no recommended bet. The first forecast for each sport/event is retained permanently. Later retraining cannot rewrite its probability or model version.')
    summary = forecast_store.summary()
    if summary.empty:
        st.info('Analyze upcoming games and collect final results to measure prospective performance.')
    else:
        st.dataframe(summary.loc[summary.sport.eq(sport)], hide_index=True)
    forecasts = forecast_store.rows()
    if not forecasts.empty:
        selected_forecasts = forecasts.loc[forecasts.sport.eq(sport)]
        st.dataframe(selected_forecasts, hide_index=True)
        st.download_button('Export prospective forecasts', selected_forecasts.to_csv(index=False), 'forecasts.csv')
    st.caption('Accuracy alone does not measure betting value. Brier measures probability error; lower is better. Fewer than 100 graded games is marked collecting evidence. This threshold is a monitoring rule, not statistical proof. Automatic model promotion and calibration are not enabled yet.')
