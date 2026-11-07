"""Weekly, time-aligned college model research; never changes live predictions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import brier_score_loss, log_loss
from sports_lab.college import forecast, COLUMNS
from sports_lab.core import utc, now_utc
from sports_lab.validation import reliability

PROTOCOL='weekly_walk_forward_v2'


def design(frame):
    return [{f'team:{r.home}':1.,f'team:{r.away}':-1.,'home_advantage':0. if r.neutral else 1.}
            for r in frame.itertuples()]


def week_cutoff(stamp):
    local=utc(stamp).tz_convert('America/Chicago')
    midnight=local.normalize()-pd.DateOffset(days=local.weekday())
    return midnight.tz_convert('UTC')-pd.Timedelta(microseconds=1)


def model_metrics(frame):
    metrics={}
    for label,column in [('elo','probability'),('margin_candidate','calibrated_probability'),('home_baseline','baseline_probability')]:
        p=frame[column].to_numpy(); y=frame.home_win.to_numpy()
        metrics[label+'_brier']=float(brier_score_loss(y,p))
        metrics[label+'_log_loss']=float(log_loss(y,p,labels=[0,1]))
        metrics[label+'_accuracy']=float(np.mean((p>=.5)==y))
    metrics['candidate_margin_mae']=float(np.mean(abs(frame.predicted_margin-(frame.home_score-frame.away_score))))
    return metrics


def compare_models(history,sport,now=None):
    if sport not in ['NCAAB','NCAAF']:
        raise ValueError('Choose NCAAB or NCAAF.')
    now=utc(now) if now is not None else now_utc()
    h=history.copy()
    missing=set(COLUMNS)-set(h)
    if missing:
        raise ValueError('Missing college history columns: '+', '.join(sorted(missing)))
    h['event_id']=h.event_id.astype(str)
    h['commence_time']=h.commence_time.map(utc)
    h['available_at']=h.commence_time+pd.Timedelta(days=1)
    h=h.loc[h.completed & h.home_score.notna() & h.away_score.notna() & h.home_score.ne(h.away_score) & (h.available_at<now)]
    h=h.sort_values(['commence_time','event_id']).drop_duplicates('event_id').copy()
    if h.empty or not np.isfinite(h[['home_score','away_score']].to_numpy(dtype=float)).all():
        raise ValueError('Supply completed games with finite scores.')
    h['season']=h.commence_time.map(lambda t:t.year if t.month>=7 else t.year-1)
    h['week_cutoff']=h.commence_time.map(week_cutoff)
    seasons=sorted(h.season.unique())
    if len(seasons)<3:
        raise ValueError('Need at least three competitive seasons for training, out-of-sample calibration warm-up, and later testing.')
    rows=[];weeks=[];skipped=[];oof=[]
    for cutoff,week in h.groupby('week_cutoff',sort=True):
        # First season supplies training; second supplies earlier out-of-sample predictions.
        if int(week.season.max())<seasons[1]:
            continue
        train=h.loc[(h.available_at<cutoff) & (h.commence_time>=cutoff-pd.Timedelta(days=730))]
        evaluating=int(week.season.min())>=seasons[2]
        if len(train)<100:
            if evaluating:
                skipped.append(dict(week_cutoff=cutoff.isoformat(),games=len(week),reason='Fewer than 100 available training games'))
            continue
        encoder=DictVectorizer(sparse=True)
        x=encoder.fit_transform(design(train))
        cap=40 if sport=='NCAAB' else 45
        age=(cutoff-train.commence_time).dt.total_seconds()/86400
        model=Ridge(alpha=10.,fit_intercept=False,solver='lsqr').fit(x,
            (train.home_score-train.away_score).clip(-cap,cap),sample_weight=np.exp2(-age/365.))
        predicted_margin=model.predict(encoder.transform(design(week)))
        counts=pd.concat([train.home,train.away]).value_counts()
        home_count=week.home.map(counts).fillna(0).astype(int)
        away_count=week.away.map(counts).fillna(0).astype(int)
        coverage=np.where(np.minimum(home_count,away_count)>=10,'Established history',
            np.where(np.minimum(home_count,away_count)==0,'Unseen participant','Limited history (1–9 games)'))
        # Past predictions are never recomputed using the current week's fitted model.
        calibration=pd.DataFrame(oof)
        if not calibration.empty:
            calibration=calibration.loc[(calibration.available_at<cutoff) &
                (calibration.commence_time>=cutoff-pd.Timedelta(days=365)) & calibration.coverage.eq('Established history')]
        can_calibrate=(len(calibration)>=50 and calibration.home_win.nunique()==2)
        if evaluating and can_calibrate:
            calibrator=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000))
            calibrator.fit(calibration[['predicted_margin']],calibration.home_win)
            candidate=calibrator.predict_proba(pd.DataFrame({'predicted_margin':predicted_margin}))[:,1]
            # Replay Elo using precisely the same available history and freeze it for this week.
            elo_forecasts,_=forecast(train,week[['event_id','home','away','commence_time','neutral']],sport,now=cutoff)
            elo_prob=elo_forecasts.set_index('event_id').home_probability
            output=week[['event_id','home','away','commence_time','home_score','away_score','season']].copy()
            output['home_win']=(week.home_score>week.away_score).astype(int)
            output['week_cutoff']=cutoff
            output['probability']=week.event_id.map(elo_prob)
            output['calibrated_probability']=candidate
            output['predicted_margin']=predicted_margin
            output['baseline_probability']=float((train.home_score>train.away_score).mean())
            output['home_history_games']=home_count
            output['away_history_games']=away_count
            output['unseen_participant']=(np.minimum(home_count,away_count)==0)
            output['coverage']=coverage
            rows.append(output)
            weeks.append(dict(week_cutoff=cutoff.isoformat(),season=int(week.season.min()),training_games=len(train),
                calibration_games=len(calibration),test_games=len(week),
                latest_training_result=train.available_at.max().isoformat(),
                latest_calibration_result=calibration.available_at.max().isoformat(),
                latest_calibration_prediction_cutoff=calibration.prediction_cutoff.max().isoformat(),
                first_test_start=week.commence_time.min().isoformat(),
                **model_metrics(output)))
        elif evaluating:
            skipped.append(dict(week_cutoff=cutoff.isoformat(),games=len(week),reason='Need 50 past out-of-sample, established-team predictions and both outcome classes'))
        # Store this week's genuine out-of-sample margins for future calibration only.
        for position,r in enumerate(week.itertuples()):
            oof.append(dict(event_id=r.event_id,commence_time=r.commence_time,available_at=r.available_at,
                prediction_cutoff=cutoff,latest_training_result=train.available_at.max(),predicted_margin=float(predicted_margin[position]),
                home_win=int(r.home_score>r.away_score),coverage=str(coverage[position])))
    if not rows:
        raise ValueError('No eligible weekly comparisons. More historical games and out-of-sample calibration warm-up are needed.')
    predictions=pd.concat(rows,ignore_index=True)
    seasonal=[];coverage_rows=[]
    for season,group in predictions.groupby('season'):
        seasonal.append(dict(season=int(season),test_games=len(group),test_weeks=int(group.week_cutoff.nunique()),
            unseen_participant_games=int(group.unseen_participant.sum()),**model_metrics(group)))
        for coverage,subset in group.groupby('coverage'):
            coverage_rows.append(dict(season=int(season),coverage=coverage,test_games=len(subset),**model_metrics(subset)))
    bins=pd.concat([reliability(group).assign(season=int(season),coverage=str(coverage))
        for (season,coverage),group in predictions.groupby(['season','coverage'])],ignore_index=True)
    bins['model']=bins.model.replace({'Uncalibrated':'Weekly frozen Elo','Calibrated candidate':'Weekly margin + OOF sigmoid'})
    summary=dict(protocol=PROTOCOL,sport=sport,generated_at=now.isoformat(),evaluated_games=len(predictions),
        folds=len(seasonal),test_weeks=len(weeks),latest_input_result=h.available_at.max().isoformat(),
        elo_brier=float(brier_score_loss(predictions.home_win,predictions.probability)),
        candidate_brier=float(brier_score_loss(predictions.home_win,predictions.calibrated_probability)),
        baseline_brier=float(brier_score_loss(predictions.home_win,predictions.baseline_probability)),
        input_sha256=hashlib.sha256(h[COLUMNS].to_json(date_format='iso').encode()).hexdigest(),skipped_weeks=skipped,
        limitations=['Research only. No live model promotion or changes to saved picks.',
            'Both models freeze at Monday 00:00 America/Chicago minus one microsecond and use the same prior 730-day game history.',
            'Candidate calibration uses only earlier weekly out-of-sample margins with results available before the cutoff; 365-day window, established teams only, at least 50 games.',
            'Training defaults remain fixed: Ridge alpha 10, recency half-life 365 days, capped margins; sigmoid C=1. No tuning on these test periods.',
            'First season supplies training; second season generates calibration predictions; reporting starts in the third competitive season.',
            'Coverage groups require 10 available prior games per team for established history. Missing-team games remain in overall results and are reported separately.',
            'No possession statistics, injuries, transfers, or player projections. No historical sportsbook benchmark or betting ROI.',
            'Historical publication times are conservatively reconstructed; source corrections remain possible. Reviewed periods become development data.',
            'Weekly Elo freezes within the test week, unlike the live game-by-game algorithm. Old seasonal and new weekly scores are not directly comparable.'])
    return dict(summary=summary,seasons=pd.DataFrame(seasonal),weeks=pd.DataFrame(weeks),predictions=predictions,
        calibration_predictions=pd.DataFrame(oof),coverage=pd.DataFrame(coverage_rows),reliability_by_season=bins)


def save_comparison(report,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    (directory/'summary.json').write_text(json.dumps(report['summary'],indent=2))
    for name in ['seasons','weeks','predictions','calibration_predictions','coverage','reliability_by_season']:
        report[name].to_csv(directory/f'{name}.csv',index=False)
