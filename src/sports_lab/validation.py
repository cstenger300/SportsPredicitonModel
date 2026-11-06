"""Expanding-season research validation with a separate past calibration season."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sports_lab.core import METRICS, validate_games, now_utc, utc


def reliability(predictions):
    rows = []
    for name, column in [('Uncalibrated','probability'),('Calibrated candidate','calibrated_probability')]:
        for bucket in range(10):
            lower, upper = bucket/10, (bucket+1)/10
            group = predictions.loc[(predictions[column]>=lower) & ((predictions[column]<upper) if upper<.999 else (predictions[column]<=1))]
            if group.empty:
                continue
            observed = float(group.home_win.mean())
            n = len(group)
            # Wilson interval for the empirical home win rate; games need not be independent.
            z = 1.96
            center = (observed+z*z/(2*n))/(1+z*z/n)
            half = z*np.sqrt(observed*(1-observed)/n+z*z/(4*n*n))/(1+z*z/n)
            rows.append(dict(model=name,probability_range=f'{lower:.0%}–{upper:.0%}',games=n,
                mean_probability=float(group[column].mean()),observed_home_win_rate=observed,
                interval_low=center-half,interval_high=center+half,
                calibration_gap_pp=100*(observed-float(group[column].mean())),
                sample_note="Small sample: fewer than 30 games" if n<30 else "Review interval and season context"))
    return pd.DataFrame(rows)


def validate_history(history, sport, now=None):
    now = utc(now) if now is not None else now_utc()
    h = validate_games(history,sport,True)
    h = h.loc[h.result_available_at < now].copy()
    # NFL season spans January; retain source season instead of using calendar year.
    if 'season' in h:
        season = pd.to_numeric(h.season,errors='raise')
        if not np.isfinite(season).all() or not (season==season.astype(int)).all():
            raise ValueError('Season must contain whole-number years.')
        h['evaluation_season'] = season.astype(int)
    else:
        h['evaluation_season'] = h.commence_time.dt.year
    seasons = sorted(h.evaluation_season.unique())
    if len(seasons)<3:
        raise ValueError('At least three seasons are needed: training, calibration, and later evaluation.')
    folds, predictions, skipped = [], [], []
    features = METRICS[sport]
    for i in range(2,len(seasons)):
        test_season, calibration_season = seasons[i],seasons[i-1]
        test = h.loc[h.evaluation_season.eq(test_season)]
        cutoff = test.metrics_as_of.min()
        calibration = h.loc[h.evaluation_season.eq(calibration_season) & (h.result_available_at < cutoff)]
        if calibration.empty:
            skipped.append(dict(season=int(test_season),reason='No calibration results available before test cutoff'))
            continue
        training = h.loc[(h.evaluation_season < calibration_season) & (h.result_available_at < calibration.metrics_as_of.min())]
        if len(training)<100 or len(calibration)<50 or training.home_win.nunique()<2 or calibration.home_win.nunique()<2:
            skipped.append(dict(season=int(test_season),reason='Need 100 training games, 50 calibration games, and both outcome classes'))
            continue
        estimator = make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=1000))
        estimator.fit(training[features],training.home_win)
        # The calibrator only sees scores from the frozen base model on the preceding season.
        calibrator = LogisticRegression(C=1.0,max_iter=1000)
        calibrator.fit(estimator.decision_function(calibration[features]).reshape(-1,1),calibration.home_win)
        raw = estimator.predict_proba(test[features])[:,1]
        calibrated = calibrator.predict_proba(estimator.decision_function(test[features]).reshape(-1,1))[:,1]
        baseline = float(training.home_win.mean())
        fold = dict(season=int(test_season),calibration_season=int(calibration_season),
            training_games=len(training),calibration_games=len(calibration),test_games=len(test),
            latest_training_result=training.result_available_at.max().isoformat(),
            first_calibration_input=calibration.metrics_as_of.min().isoformat(),
            latest_calibration_result=calibration.result_available_at.max().isoformat(),
            first_test_input=cutoff.isoformat())
        for label,p in [('raw',raw),('calibrated',calibrated),('baseline',np.full(len(test),baseline))]:
            fold[label+'_brier']=float(brier_score_loss(test.home_win,p))
            fold[label+'_log_loss']=float(log_loss(test.home_win,p,labels=[0,1]))
            fold[label+'_accuracy']=float(np.mean((p>=.5)==test.home_win))
        folds.append(fold)
        output = test[['event_id','home','away','commence_time','home_win']].copy()
        output['season']=test_season
        output['probability']=raw
        output['calibrated_probability']=calibrated
        output['baseline_probability']=baseline
        output['model_direction']=np.where(raw>=.5,'Model favors home','Model favors away')
        predictions.append(output)
    if not predictions:
        raise ValueError('No valid chronological folds. Supply more seasons and completed games.')
    combined = pd.concat(predictions,ignore_index=True)
    by_direction = []
    for direction,group in combined.groupby('model_direction'):
        by_direction.append(dict(group=direction,games=len(group),
            accuracy=float(((group.probability>=.5)==group.home_win).mean()),
            brier=float(brier_score_loss(group.home_win,group.probability))))
    summary = dict(sport=sport,generated_at=now.isoformat(),evaluated_games=len(combined),folds=len(folds),
        latest_input_result=h.result_available_at.max().isoformat(),
        input_sha256=hashlib.sha256(h.to_json(date_format='iso').encode()).hexdigest(),
        raw_brier=float(brier_score_loss(combined.home_win,combined.probability)),
        calibrated_brier=float(brier_score_loss(combined.home_win,combined.calibrated_probability)),
        baseline_brier=float(brier_score_loss(combined.home_win,combined.baseline_probability)),
        skipped_folds=skipped,
        limitations=['Research report only: no changes to live models, probabilities, or saved picks.',
          'Each base model is frozen before its calibration season, then evaluated on the following season.',
          'Historical source corrections and reconstructed publication times limit point-in-time claims.',
          'The constant home-win baseline is not a sportsbook benchmark; no historical odds were supplied.',
          'Calibration candidate was fixed in advance; inspecting this report makes these periods development data.',
          'Reliability intervals assume independent observations; repeated teams and changing seasons can invalidate that assumption.',
          'Calendar-year folds are used only when a season column is unavailable.'])
    seasonal_reliability = pd.concat([reliability(group).assign(season=int(season))
        for season,group in combined.groupby('season')],ignore_index=True)
    return dict(summary=summary,seasons=pd.DataFrame(folds),predictions=combined,
                reliability_by_season=seasonal_reliability,
                reliability=reliability(combined),directions=pd.DataFrame(by_direction))


def save_report(report, directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'summary.json').write_text(json.dumps(report['summary'],indent=2))
    for name in ['seasons','predictions','reliability','reliability_by_season','directions']:
        report[name].to_csv(directory/f'{name}.csv',index=False)
