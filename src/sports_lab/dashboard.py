"""Multi-sport research dashboard."""
from pathlib import Path
import pandas as pd
import streamlit as st
from sports_lab.core import METRICS, Ledger, fit_model, rank_quotes
from sports_lab import provider
from sports_lab.performance import ForecastStore
from sports_lab.recommendations import summarize_quotes
from sports_lab.shadow import ShadowStore
from sports_lab.collection import CollectionStore, coverage_table, SPORTS
st.set_page_config(page_title='Sports Edge Lab', page_icon='📊', layout='wide')
st.title('Sports Edge Lab')
st.caption('NBA · NHL · NFL · NCAAB · NCAAF · Tennis | Advanced metrics · Sportsbook comparison · Paper results')
sport = st.sidebar.selectbox('Sport', list(METRICS))
st.sidebar.info('Moneylines only. Research probabilities are not calibrated confidence levels. No real wagers are placed.')
ledger = Ledger(Path(__file__).resolve().parents[2] / 'data/sports_lab/paper.sqlite')
forecast_store = ForecastStore(ledger.path)
shadow_store = ShadowStore(ledger.path)
collection_store = CollectionStore(ledger.path.parent / 'collections')
board, tracker, performance_tab, collection_tab, help_tab = st.tabs(['Find value', 'Results tracker', 'Model performance', 'Data collection', 'Data and methodology'])
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
            events,quotes=provider.odds(key)
            st.session_state[f'live_{sport}']=(events,quotes)
            try:
                collection_store.archive_odds(sport,key,events,quotes)
            except ValueError as archive_error:
                st.warning(str(archive_error))
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
                    try:
                        collection_store.archive_metrics('NFL',h,u,manifest)
                    except ValueError as archive_error:
                        st.warning(str(archive_error))
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
    if sport in ['NCAAB','NCAAF']:
        st.subheader('Automatic public college data · experimental Elo baseline')
        college_date = st.date_input('College game date', pd.Timestamp.now(tz='America/Chicago').date(), key='college_date_'+sport)
        st.warning('This first college model uses game results and Elo ratings, not the advanced-metric CSV model. It does not include injuries, transfers, current starters, or adjusted efficiency/EPA. Offseason ratings shrink toward average. Use only for paper workflow testing.')
        st.caption('Downloads about 18 months of public ESPN results with local caching. First run may take several minutes. Names and start times must match exactly for sportsbook comparisons; unmatched events are skipped.')
        if st.button('Show '+sport+' schedule for selected date'):
            try:
                from sports_lab.college import fetch_games
                schedule = fetch_games(sport,college_date,college_date,ledger.path.parents[2]/'data/sports_lab/college',refresh=True)
                if schedule.empty:
                    st.info('No games returned for this date.')
                else:
                    schedule['Start · Central']=pd.to_datetime(schedule.commence_time,utc=True).dt.tz_convert('America/Chicago').dt.strftime('%b %d %I:%M %p %Z')
                    schedule['State']=schedule.apply(lambda r: 'Final' if r.completed else 'Started / status not final' if pd.Timestamp(r.commence_time)<=pd.Timestamp.now(tz='UTC') else 'Upcoming',axis=1)
                    st.dataframe(schedule[['home','away','Start · Central','State']],hide_index=True)
            except (ValueError,KeyError,OSError) as exc:
                st.error(str(exc))
        if st.button('Download '+sport+' data and analyze',type='primary'):
            try:
                from sports_lab.college import load, forecast, align_events, save_mapping
                with st.spinner('Downloading college results and evaluating the Elo baseline…'):
                    h,u = load(ledger.path.parents[2],sport,college_date)
                    q = pd.DataFrame()
                    if live is not None and not live[0].empty:
                        total=len(u)
                        u,q=align_events(u,live[0],live[1])
                        save_mapping(ledger.path.parents[2],sport,u)
                        st.caption(f'Matched {len(u)} of {total} public games to sportsbook events. Unmatched games were not priced.')
                    forecasts,report=forecast(h,u,sport)
                    ranked=rank_quotes(forecasts,q,report,sport) if not q.empty else pd.DataFrame()
                    forecast_store.record(forecasts,report,sport)
                    st.session_state['analysis_'+sport]=(forecasts,report,ranked)
                    st.success('College baseline forecasts ready. These are not validated advanced-metric recommendations.')
            except (ValueError,KeyError,OSError) as exc:
                st.error(str(exc))
    if sport in ['NCAAB','NCAAF']:
        st.subheader('Prospective side-by-side test · no bets')
        st.caption('Save weekly Elo and margin-candidate probabilities for all upcoming matchups with at least 10 prior games per team. This is separate from paper picks and does not select wagers. No odds key is needed. A completed college research-history download is required.')
        if st.button('Record both models for upcoming '+sport+' games'):
            try:
                from sports_lab.college import load
                from sports_lab.shadow import predict_pair
                root=ledger.path.parents[2]
                research_file=root/'data/sports_lab/college'/f'{sport}_research_history.csv'
                if not research_file.exists():
                    raise ValueError('Run the college weekly research comparison first to download the extended history.')
                with st.spinner('Refreshing public results and generating the paired pregame snapshot…'):
                    recent,upcoming_games=load(root,sport,college_date)
                    combined=pd.concat([pd.read_csv(research_file),recent],ignore_index=True)
                    combined['event_id']=combined.event_id.astype(str)
                    combined=combined.drop_duplicates('event_id',keep='last')
                    paired=predict_pair(combined,upcoming_games,sport)
                    added=shadow_store.record(paired)
                skipped=int(paired.coverage.ne('Established history').sum())
                flagged=int(paired.data_quality_review.sum())
                st.success(f'{added} new paired game forecasts saved. History check flagged {skipped} matchups; schedule review flagged {flagged}. Existing snapshots were preserved.')
                st.dataframe(paired[['home','away','commence_time','elo_home_probability','margin_home_probability','coverage','data_quality_review','data_quality_reason']],hide_index=True)
            except (ValueError,KeyError,OSError) as exc:
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
        if report.get('model_type'):
            st.warning(report['model_type'])
            st.caption(report.get('limitations',''))
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
    st.caption('College sports use public ESPN finals; other sports use The Odds API quota. Checks the selected sport/tournament while open. Results older than three days may require reviewed CSV imports.')
    @st.fragment(run_every='15m' if auto_scores else None)
    def refresh_scores_automatically():
        if auto_scores and key:
            try:
                if sport in ['NCAAB','NCAAF']:
                    from sports_lab.college import final_results
                    results=final_results(ledger.path.parents[2],sport)
                else:
                    results = provider.scores(sport, key)
                forecast_store.settle(results)
                shadow_count=shadow_store.settle(results)
                count = ledger.settle(results)
                st.caption(f'Latest automatic check: {pd.Timestamp.now(tz="America/Chicago").strftime("%b %d %I:%M %p %Z")} · {count} newly settled picks, {shadow_count} paired forecasts. Reopen the table to reload results.')
            except ValueError as exc:
                st.warning(str(exc))
    refresh_scores_automatically()
    st.subheader('Paper results · original prices preserved')
    if key and st.button('Refresh final scores'):
        try:
            results = provider.scores(sport,key)
            forecast_store.settle(results)
            shadow_count=shadow_store.settle(results)
            st.success(f'{ledger.settle(results)} picks settled.')
        except ValueError as e:
            st.error(str(e))
    if sport in ['NCAAB','NCAAF'] and st.button('Refresh college finals from public data'):
        try:
            from sports_lab.college import final_results
            results=final_results(ledger.path.parents[2],sport)
            forecast_store.settle(results)
            shadow_count=shadow_store.settle(results)
            st.success(f'{ledger.settle(results)} college paper picks and {shadow_count} paired forecasts settled.')
        except (ValueError,KeyError,OSError) as exc:
            st.error(str(exc))
    uploaded = st.file_uploader('Reviewed final results CSV', type='csv')
    if uploaded and st.button('Settle imported results'):
        try:
            results = pd.read_csv(uploaded)
            forecast_store.settle(results)
            shadow_count=shadow_store.settle(results)
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
            st.write('Calibration by season')
            season_bins = result['reliability_by_season']
            chosen_season = st.selectbox('Calibration season to inspect', sorted(season_bins.season.unique()), key='calibration_season')
            st.dataframe(season_bins.loc[season_bins.season.eq(chosen_season)],hide_index=True)
            st.caption('Calibration gap is observed minus predicted probability in percentage points. Negative means overprediction in that group. Fewer than 30 games is flagged for context; it is not a significance test. Do not reverse picks or tune a strategy to these bins.')
            st.write('Focus: 50–60% predicted home win probability')
            st.dataframe(season_bins.loc[season_bins.probability_range.eq('50%–60%')],hide_index=True)
            st.write('Breakdown by model direction · not sportsbook favorites/underdogs')
            st.dataframe(result['directions'],hide_index=True)
            for limitation in summary['limitations']:
                st.caption(limitation)
            for name in ['seasons','predictions','reliability','reliability_by_season','directions']:
                st.download_button('Export validation '+name,result[name].to_csv(index=False),f'validation_{name}.csv')
    if sport in ['NCAAB','NCAAF']:
        st.divider()
        st.subheader('Paired prospective forecasts')
        pairs=shadow_store.rows()
        if pairs.empty or pairs.loc[pairs.sport.eq(sport)].empty:
            st.info('In Find value, select a future college game date and click Record both models for upcoming games. No paper pick or odds download is required.')
        else:
            pairs=pairs.loc[pairs.sport.eq(sport)]
            st.caption(f"{len(pairs)} paired games saved · {pairs.winner.isna().sum()} pending. Original probabilities cannot be overwritten. These are forecast results, not a betting ROI record.")
            st.dataframe(pairs[['home','away','commence_time','elo_home_probability','margin_home_probability','status','elo_brier','margin_brier','data_quality_review','data_quality_reason']],hide_index=True)
            flagged=int(pairs.data_quality_review.sum())
            if flagged:
                st.warning(f'{flagged} saved games require schedule/identity review. The summary below excludes them; original forecasts and results remain in the export. This is a review flag, not a declaration that a game is invalid.')
            if st.button('Audit saved '+sport+' schedules against public source'):
                try:
                    from sports_lab.college import fetch_games
                    source_rows=[]
                    days=sorted(pd.to_datetime(pairs.commence_time,utc=True).dt.tz_convert('America/New_York').dt.date.unique())
                    for day in days:
                        source_rows.append(fetch_games(sport,day,day,ledger.path.parents[2]/'data/sports_lab/college',refresh=True))
                    source=pd.concat(source_rows,ignore_index=True)
                    source['sport']=sport
                    issues=shadow_store.audit_saved(source)
                    st.dataframe(issues,hide_index=True)
                    st.caption('This rechecks ESPN source identities and schedules. Repeated source data is not independent verification. Existing flags are retained until investigated; no original record is rewritten.')
                    st.dataframe(source[['event_id','home','home_id','away','away_id','commence_time','completed','home_score','away_score']],hide_index=True)
                    st.download_button('Export source schedule audit',source.to_csv(index=False),sport+'_source_schedule_audit.csv')
                except (ValueError,KeyError,OSError) as exc:
                    st.error(str(exc))
            paired_summary=shadow_store.summary(sport)
            if not paired_summary.empty:
                st.dataframe(paired_summary,hide_index=True)
            st.caption('After games finish, open Results tracker and click Refresh college finals from public data. Shadow forecasts use public event IDs. Model versions, input cutoffs, and original probabilities are preserved. Small samples are exploratory; no model is automatically promoted.')
            st.download_button('Export paired forecasts',pairs.to_csv(index=False),sport+'_paired_forecasts.csv')
    if sport in ['NCAAB','NCAAF']:
        st.divider()
        st.subheader('College team-strength research · no live model changes')
        st.write('Weekly walk-forward comparison: both models freeze before each test week using the same available history. The margin candidate refreshes team strength weekly and learns its probability mapping only from earlier out-of-sample predictions. Possession efficiency and player availability are not included.')
        st.caption('This comparison requires multiple seasons. Downloading three completed seasons plus the current season can take several minutes. Existing cached requests are reused. No odds requests are made by this research panel.')
        reuse_history=st.checkbox('Reuse downloaded college research history',value=True,key='reuse_college_history_'+sport)
        st.caption('Leave reuse enabled to compare the same cached input as your previous report. Uncheck to refresh history; no paid odds calls are made.')
        if st.button('Run '+sport+' weekly research comparison'):
            try:
                from sports_lab.college import load_history
                from sports_lab.college_research import compare_models,save_comparison
                root=ledger.path.parents[2]
                today=pd.Timestamp.now(tz='America/Chicago').normalize().tz_localize(None)
                season_year=today.year if today.month>=7 else today.year-1
                with st.spinner('Downloading seasonal history and comparing team-strength models…'):
                    history_path=root/'data/sports_lab/college'/f'{sport}_research_history.csv'
                    if reuse_history and history_path.exists():
                        research_history=pd.read_csv(history_path)
                    else:
                        research_history=load_history(root,sport,pd.Timestamp(year=season_year-3,month=7,day=1),today-pd.Timedelta(days=1))
                        research_history.to_csv(history_path,index=False)
                    comparison=compare_models(research_history,sport)
                    save_comparison(comparison,root/'artifacts/sports_lab'/f'{sport}_team_strength_weekly_v2')
                    st.session_state['college_research_v2_'+sport]=comparison
            except (ValueError,KeyError,OSError) as exc:
                st.error(str(exc))
        comparison=st.session_state.get('college_research_v2_'+sport)
        if comparison:
            stats=comparison['summary']
            st.caption(f"{stats['evaluated_games']} evaluated games across {stats['test_weeks']} frozen test weeks and {stats['folds']} seasons · {stats['generated_at']}")
            st.caption('Latest available result in cached input: '+stats['latest_input_result'])
            if stats['skipped_weeks']:
                with st.expander('Weeks omitted because training/calibration history was insufficient'):
                    st.dataframe(pd.DataFrame(stats['skipped_weeks']),hide_index=True)
            c=st.columns(3)
            c[0].metric('Weekly frozen Elo Brier',f"{stats['elo_brier']:.3f}")
            c[1].metric('Margin candidate Brier',f"{stats['candidate_brier']:.3f}")
            c[2].metric('Home-win baseline Brier',f"{stats['baseline_brier']:.3f}")
            st.info('Lower Brier is better. Even if the candidate improves, no live model is automatically replaced. We must review seasonal results, log loss, coverage, and calibration before considering promotion.')
            st.dataframe(comparison['seasons'],hide_index=True)
            st.write('Coverage: both models evaluated on identical games within each group')
            st.dataframe(comparison['coverage'],hide_index=True)
            with st.expander('Weekly training and calibration cutoffs'):
                st.dataframe(comparison['weeks'],hide_index=True)
            seasonal=comparison['reliability_by_season']
            selected_season=st.selectbox('College calibration season',sorted(seasonal.season.unique()),key='college_research_season_'+sport)
            st.dataframe(seasonal.loc[seasonal.season.eq(selected_season)],hide_index=True)
            for limitation in stats['limitations']:
                st.caption(limitation)
            for name in ['seasons','weeks','predictions','coverage','calibration_predictions','reliability_by_season']:
                st.download_button('Export college '+name,comparison[name].to_csv(index=False),sport+'_college_'+name+'.csv')

with collection_tab:
    st.subheader('Multi-sport data collection')
    st.write('Archive data now while models are developed separately. Successful collection does not establish model readiness. Snapshots retain retrieval times and are never rewritten.')
    st.dataframe(coverage_table(),hide_index=True)
    chosen_sports=st.multiselect('Sports to collect',SPORTS,default=['NBA','NHL','NFL','NCAAB','NCAAF'])
    collection_date=st.date_input('Public schedule date',pd.Timestamp.now(tz='America/Chicago').date(),key='collection_date')
    public_enabled=st.checkbox('Collect public schedules and results',value=True)
    import os
    key_present=bool(os.environ.get('ODDS_API_KEY'))
    st.caption('Odds API key detected.' if key_present else 'No Odds API key detected. Public collection works without it.')
    odds_enabled=st.checkbox('Archive current sportsbook odds · uses API quota',value=False,disabled=not key_present)
    st.caption('Odds collection requests one US-region two-way moneyline feed per selected sport/tournament per cycle. Provider credits depend on your plan and response; HTTP request counts are not a credit guarantee. Odds snapshots include returned upcoming dates, not only the public schedule date.')
    if 'Tennis' in chosen_sports:
        if st.button('Load active tennis tournaments for collection'):
            try:
                st.session_state['collection_tennis_keys']=provider.tournaments()
            except ValueError as exc:
                st.error(str(exc))
        tennis_options=st.session_state.get('collection_tennis_keys',[])
        collection_tennis_key=st.selectbox('Tennis tournament to archive',tennis_options) if tennis_options else None
        st.caption('One active tournament per collection cycle. Public tennis schedules and advanced metrics are not connected yet.')
    else:
        collection_tennis_key=None
    interval=st.selectbox('Collection interval / per-feed cooldown in minutes',[15,30,60,120],index=2)
    auto_collect=st.checkbox('Collect periodically while this app session is open',value=False)
    if odds_enabled:
        odds_calls=sum(s!='Tennis' or collection_tennis_key is not None for s in chosen_sports)
        st.caption(f'At these settings: up to {odds_calls} odds requests per cycle, about {odds_calls*60/interval:g} per hour while periodic collection is active. Loading tennis tournament metadata is a separate request.')
    st.caption('Recurring odds calls occur only when both odds archiving and periodic collection are enabled. No collection runs while the app is closed. Recent manual price snapshots also trigger the collector cooldown.')
    def run_selected_collection():
        if not chosen_sports or not (public_enabled or odds_enabled):
            st.warning('Choose at least one sport and one source.')
            return
        with st.spinner('Collecting selected sources and saving snapshots…'):
            result=collection_store.collect(chosen_sports,collection_date,include_public=public_enabled,
                include_odds=odds_enabled,tennis_key=collection_tennis_key,cooldown_minutes=interval)
        st.dataframe(result,hide_index=True)
    if st.button('Collect selected feeds now'):
        run_selected_collection()
    @st.fragment(run_every=f'{interval}m' if auto_collect else None)
    def show_collection_status():
        if auto_collect:
            run_selected_collection()
        status=collection_store.status()
        for column in ['last_attempt','last_success']:
            status[column]=pd.to_datetime(status[column],utc=True,format='mixed').dt.tz_convert('America/Chicago').dt.strftime('%b %d %Y %I:%M %p %Z')
        st.write('Observed feed status · latest attempt and last successful snapshot')
        st.dataframe(status,hide_index=True)
        st.download_button('Export collection status',status.to_csv(index=False),'collection_status.csv')
        runs=collection_store.runs()
        if not runs.empty:
            with st.expander('Collection run history and snapshot hashes'):
                st.dataframe(runs,hide_index=True)
                st.download_button('Export collection runs',runs.to_csv(index=False),'collection_runs.csv')
        st.caption('Archives: data/sports_lab/collections/snapshots. Preserve this directory and runs.sqlite when moving computers. Source errors are shown explicitly; the last successful snapshot remains available. Advanced-statistics and player-availability feeds are still missing for most sports.')
    show_collection_status()
