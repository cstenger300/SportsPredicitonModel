# Sports Edge Lab — Windows desktop setup

Extract the complete project ZIP into a normal user folder, such as Documents.
Do not run from inside the ZIP or from Windows/System32. Install 64-bit Python
3.12, then double-click **Start-Sports-Lab.cmd** in the extracted project folder.
It installs pinned dependencies into a local .venv and opens the local dashboard
in your browser. Keep its terminal open. Close it or press Ctrl+C to stop.
The launcher does not require Administrator access. Windows execution has not
been tested on a Windows machine in this development session.

The app runs locally; API requests still require Internet access. Set ODDS_API_KEY
in Windows environment-variable settings, then reopen the launcher. Cloud secrets
do not transfer to your computer. Never put the key in chat, Git, or CSV files.

## First workflow

1. Choose NBA, NHL, NFL, NCAAB, or Tennis.
2. Open Data and methodology and download metric templates. Supply at least 100
   historical games with authentic pregame metrics and result availability times.
   Historical advanced-statistics feeds have not yet been integrated; imports
   are required. The Odds API supplies prices and supported scores, not these metrics.
3. Fetch prices (select a tournament first for tennis), export event IDs, and
   match your upcoming metrics to those participants and start times.
4. Analyze matchups. The first pregame forecast for every event is saved in SQLite,
   even if no price is available or no bet qualifies. Refitting on updated history
   creates a new model version; it cannot rewrite past forecasts.
5. Record a qualifying one-unit paper pick to preserve its exact original odds.
6. In Results tracker, refresh final scores or enable 15-minute automatic checks
   while the app is open. Automatic checks cover the selected sport/tournament
   and consume quota. Reload the tracker table to see background updates. Scores
   older than three days, unsupported sports, and exceptional book settlements
   need reviewed CSV imports. No checks run while the app is closed.
7. Model performance shows prospective accuracy, Brier score, baseline Brier,
   sample sizes, and model versions, independently of which picks were selected.

Back up **data/sports_lab/paper.sqlite**. It contains your forecasts and paper picks.
Do not replace it when updating application code. This version supports two-way
moneylines only; probabilities are uncalibrated research estimates. It does not
place wagers or guarantee profitable picks. Ties/retirements/cancellations need
settlement-rule review. Real feed access needs your key and subscription.

## Development milestones

Implemented: five sport schemas, chronological logistic-regression baseline,
price comparison, immutable paper picks, prospective forecast/version records,
manual and opt-in periodic score updates, and performance reporting.

Next: integrate licensed historical statistics and availability feeds; compare
sport-specific ratings and scoring distributions against baseline and boosting;
add walk-forward calibration and candidate/champion promotion using untouched
future periods; then expand to spread/total markets. Updating ratings, retraining,
and promoting a model are distinct operations. No automatic model promotion is
currently enabled, and the app never changes strategy just to exceed 50% accuracy.

## Automatic NFL data

Choose NFL, select a game date, and click **Download NFL data and analyze**.
No metric CSVs are required for this path. The app downloads the public nflverse
schedule and regular-season play-by-play, caches historical seasons locally,
refreshes the current season, and builds shifted rolling metrics. First use may
need approximately 100 MB or more and several minutes. Raw files are kept in
`data/raw`; source hashes and generated inputs are saved in `data/sports_lab/nfl`.

For sportsbook comparisons, fetch current prices first, then run the automatic
NFL analysis. The app matches both teams and kickoff times before using provider
IDs. No odds key is needed for the public-data forecasts alone. Models use rolling
EPA, success and explosive-play rates, defensive sack rate, rest, and opponent-
adjusted EPA ratings. Sack rate is not pressure rate. This does not yet include
confirmed starters, injuries, or calibrated probabilities. Historical source files
may contain later corrections; reconstructed input dates are not proof of actual
historical publication times. No profitable-betting claim is implied by validation.

The automatic feed was checked on current data for November 8, 2026 (13 upcoming
games) and a historical replay for December 3, 2023 (11 upcoming games). Provider
availability and future schedule changes can affect later requests. Other sports
still require metric imports. This feature fetches and computes data when requested;
it does not run unattended while the app is closed.

## Recommendation display

The main table now shows one best displayed selection per game rather than a row
for every bookmaker. Fresh qualifying quotes are preferred; other book prices
for the selected side appear under the explanation, and both sides remain in
All quotes. Prediction reasoning and price reasoning are separate. Probabilities,
break-even rates, and estimated returns are formatted as percentages.

A difference of at least 10 percentage points between model probability and a
quote's break-even probability triggers a review warning. This is a transparent
UI heuristic, not a validated confidence threshold or no-vig market comparison.
It does not prove an edge or automatically block a paper candidate. Quotes become
ineligible for recording after 15 minutes or game start; refresh and analyze again.
Existing paper-pick snapshots are unchanged. Export recommendations for the summary
or export all quotes for the full comparison.

## Historical NFL probability validation

Under **Model performance**, click **Run NFL validation report** after preparing
NFL metrics with the automatic download. At least three historical seasons are
needed. With the default 2022 start, 2022 trains the base model, 2023 fits a
sigmoid calibration candidate using the frozen model's scores, and 2024 evaluates
both. Subsequent folds expand the training history, use the preceding season for
calibration, and test the next season. Results unavailable at each input cutoff
are excluded. This is a fixed research procedure, separate from the live model's
80/20 validation and final refit; the displayed error is not an identical replay
of the live production training policy.

The report includes seasonal accuracy/Brier/log loss, a constant historical
home-win baseline, reliability bins with counts and approximate Wilson intervals,
and breakdowns by which side the model favors. Those are not sportsbook
favorite/underdog labels. The report does not infer profitability or market
superiority without historical timestamped odds. Historical corrections and
reconstructed publication timestamps still limit point-in-time claims. The
current season is partial; interval assumptions can fail with correlated games.

Reports and predictions are exported to `artifacts/sports_lab/nfl_validation`,
including input hash and latest result timestamp. Re-run after refreshing data.
No live probability, model version, or saved paper pick is changed. A candidate
calibrator is not automatically adopted even when one period improves. These
reviewed periods become development data; future promotion needs new evidence.
