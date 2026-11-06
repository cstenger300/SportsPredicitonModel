"""Multi-sport research dashboard."""
from pathlib import Path
import pandas as pd
import streamlit as st
from sports_lab.core import METRICS, Ledger, fit_model, rank_quotes
from sports_lab import provider
from sports_lab.performance import ForecastStore
from sports_lab.recommendations import summarize_quotes
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
    if sport == 'NFL':
        st.subheader('Automatic public NFL data')
        game_date = st.date_input('NFL game date', pd.Timestamp.now(tz='America/Chicago').date())
        first_season = st.number_input('First historical season', min_value=2016, max_value=2030, value=2022, step=1)
        st.caption('Downloads public nflverse play-by-play and builds rolling EPA, success, explosive-play, defensive sack-rate, rest, and opponent-adjusted metrics. Historical files can contain later corrections. Injuries and confirmed starters are not included. First download may take several minutes.')
        if st.button('Download NFL data and analyze', type='primary'):
            try:
                from sports_lab.nfl_feed import load_inputs, align_odds
                with st.spinner('Downloading NFL history and building pregame metrics…'):
                    h, u, manifest = load_inputs(ledger.path.parents[2], game_date, int(first_season), refresh=True)
                    q = pd.DataFrame()
                    if live is not None and not live[0].empty:
                        u, q = align_odds(u, live[0], live[1])
                    forecasts, report = fit_model(h, u, 'NFL')
                    report['data_source'] = 'nflverse reconstructed historical pregame metrics'
                    ranked = rank_quotes(forecasts, q, report, 'NFL') if not q.empty else pd.DataFrame()
                    forecast_store.record(forecasts, report, 'NFL')
                    st.session_state['analysis_NFL'] = (forecasts, report, ranked)
                    st.success('NFL data downloaded and analyzed. No metric CSV uploads required.')
                    st.caption('Source refresh: ' + manifest['retrieved_at'])
            except (ValueError, KeyError, OSError) as exc:
                st.error(str(exc))
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
            best = summarize_quotes(ranked)
            st.subheader('Best displayed selection per game')
            st.caption('One selection per game, ranked by estimated return. Prices are compared among fresh quotes passing the candidate checks. Model probability is not a calibrated confidence level.')
            display = best[['matchup','selection','bookmaker','american_odds','probability','break_even','edge','expected_return','status']].rename(columns={
                'matchup':'Game','selection':'Selection','bookmaker':'Sportsbook','american_odds':'American odds',
                'probability':'Model win probability','break_even':'Break-even probability','edge':'Probability difference',
                'expected_return':'Estimated return','status':'Status'})
            st.dataframe(display.style.format({'Model win probability':'{:.1%}','Break-even probability':'{:.1%}',
                'Probability difference':'{:+.1%}','Estimated return':'{:+.1%}'}), hide_index=True)
            for item in best.to_dict('records'):
                with st.expander(f"{item['selection']} · {item['bookmaker']} {item['american_odds']:+g} · Why this price?"):
                    if item['large_disagreement']:
                        st.warning('Large model–price disagreement (at least 10 percentage points versus break-even). This is a review flag, not evidence of a strong edge. Check participant matching, injuries, confirmed starters, and calibration. No market-consensus adjustment has been made.')
                    st.write('Prediction: ' + item['prediction_reason'])
                    st.write('Price: ' + item['price_reason'])
                    st.caption('Missing inputs: confirmed starters and injury adjustments. Winning probability and price value are different: an underdog can have estimated value without being the likely winner.')
                    alternatives = ranked.loc[ranked.event_id.eq(item['event_id']) & ranked.side.eq(item['side'])]
                    st.write('Other sportsbook prices for this selection')
                    st.dataframe(alternatives[['bookmaker','american_odds','observed_at','source_updated_at']],hide_index=True)
            with st.expander('All quotes · both sides'):
                st.dataframe(ranked.drop(columns='validation'),hide_index=True)
            st.download_button('Export recommendations',best.drop(columns='validation').to_csv(index=False),'recommendations.csv')
            st.download_button('Export all quotes', ranked.to_csv(index=False), 'analysis.csv')
            candidates = best.loc[best.eligible]
            if candidates.empty:
                st.info('No qualifying paper candidates in this snapshot. Refresh prices and analyze again if quotes are stale.')
            else:
                selected = st.selectbox('Track a candidate', candidates.index, format_func=lambda i: f"{best.loc[i,'selection']} · {best.loc[i,'bookmaker']} · {best.loc[i,'american_odds']:+g}")
                if st.button('Record one-unit paper pick'):
                    try:
                        added = ledger.record(best.loc[selected].to_dict())
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
    if sport == 'NFL':
        st.divider()
        st.subheader('Historical probability validation · research only')
        st.write('Evaluate later seasons using a model trained on earlier seasons and a separate preceding calibration season. This report does not update the live model or saved picks.')
        history_file = ledger.path.parents[2] / 'data/sports_lab/nfl/history.csv'
        if not history_file.exists():
            st.info('Use Download NFL data and analyze first to prepare historical metrics.')
        elif st.button('Run NFL validation report'):
            try:
                from sports_lab.validation import validate_history, save_report
                with st.spinner('Testing chronological predictions and calibration…'):
                    result = validate_history(pd.read_csv(history_file), 'NFL')
                    save_report(result, ledger.path.parents[2] / 'artifacts/sports_lab/nfl_validation')
                    st.session_state['nfl_validation'] = result
            except (ValueError, KeyError, OSError) as exc:
                st.error(str(exc))
        result = st.session_state.get('nfl_validation')
        if result:
            summary = result['summary']
            st.caption(f"Generated {summary['generated_at']} · {summary['evaluated_games']} evaluated games across {summary['folds']} seasonal folds")
            st.caption('Latest result in cached input: ' + summary['latest_input_result'])
            st.caption('Historical sportsbook comparison: unavailable until timestamped odds are supplied. No bookmaker favorite/underdog breakdown or betting ROI is inferred from these statistics.')
            if summary['calibrated_brier'] >= summary['raw_brier']:
                st.info('This calibration candidate did not improve overall Brier score. It has not been applied to live predictions.')
            cols = st.columns(3)
            cols[0].metric('Uncalibrated Brier', f"{summary['raw_brier']:.3f}")
            cols[1].metric('Calibration candidate Brier', f"{summary['calibrated_brier']:.3f}")
            cols[2].metric('Historical baseline Brier', f"{summary['baseline_brier']:.3f}")
            st.write('Season-by-season results · lower Brier and log loss are better')
            st.dataframe(result['seasons'],hide_index=True)
            st.write('Calibration: predicted home win probability versus observed home win rate')
            st.dataframe(result['reliability'],hide_index=True)
            plot = result['reliability'].pivot(index='mean_probability',columns='model',values='observed_home_win_rate')
            st.scatter_chart(plot,x_label='Mean predicted home win probability',y_label='Observed home win rate')
            st.write('Breakdown by model direction · not sportsbook favorites/underdogs')
            st.dataframe(result['directions'],hide_index=True)
            for limitation in summary['limitations']:
                st.caption(limitation)
            for name in ['seasons','predictions','reliability','directions']:
                st.download_button('Export validation '+name,result[name].to_csv(index=False),f'validation_{name}.csv')
