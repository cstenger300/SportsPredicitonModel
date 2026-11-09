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

The validation report also includes `reliability_by_season.csv`, a season selector,
and a focused 50–60% table. Calibration gap is observed minus predicted home-win
rate in percentage points. Negative gaps indicate overprediction in that group.
Cells below 30 games are flagged as small samples; 30 is a display heuristic,
not a statistical-significance threshold. Calibrated and uncalibrated bins can
contain different games, so their bin win rates are not paired comparisons.
Reviewed bins must not be used to automatically reverse selections or tune a
strategy without new out-of-sample evaluation.

## College paper testing: NCAAB and NCAAF

Both sports are now available in the Sport dropdown. NCAAF uses The Odds API key
`americanfootball_ncaaf`; NCAAB uses `basketball_ncaab`. Select a local game date,
fetch current sportsbook prices, then choose **Download NCAAB data and analyze**
or **Download NCAAF data and analyze**. **Show [sport] schedule for selected date**
checks the public schedule without first downloading history. Games already
started are not forecast. No games are fabricated when a feed returns nothing.

The automatic college path is an experimental results-only Elo baseline, separate
from the advanced-metric CSV model. It is NOT an implementation of KenPom-style
adjusted efficiencies, possession statistics, or college football EPA. Current
starters, injuries, transfers, roster continuity, and coaching changes are absent.
K=24, home-rating advantage=55 Elo points (zero at neutral sites), and 50%
offseason shrinkage are fixed exploratory defaults, not tuned validated settings.
The baseline should only be used to test the paper workflow. The final 20% of
historical games are evaluated in chronological sequence; ratings update only
after conservatively estimated result availability. Baseline comparisons use only
results available before the validation cutoff. Reports do not claim calibration
or profitable betting.

The app caches about 550 days of ESPN scoreboard results in
`data/sports_lab/college`, splits potentially truncated requests, and blocks
candidates for participants with fewer than 10 observed prior games. Names must
match after punctuation/case normalization and start times must be within 15
minutes. No fuzzy team-name guesses are made. Unmatched games are skipped or
reported explicitly. The provider may not publish all events or historical data.

For settlement, use **Refresh college finals from public data** in Results tracker
if The Odds API scores endpoint does not support your sport/subscription. Public
IDs are mapped to matched sportsbook IDs in local event-mapping CSVs; preserve
these files along with your SQLite database. This public refresh covers the last
three days and uses the provider's completed status, not the presence of scores.
Reviewed CSV imports remain available for exceptional or older results.

Source requirements: HTTPS access to `site.api.espn.com` and, for odds,
`api.the-odds-api.com`. No ESPN credential is used. A network/format failure stops
analysis instead of substituting invented data. Fixtures and UI tests pass, but
live ESPN access was blocked by the cloud network policy during development;
tonight's live availability must be checked from the Windows app. Windows users
do not need to change cloud settings. An additive ESPN allowlist requirement was
saved to the cloud draft for future validation; that save does not apply runtime
policy. Paid odds calls consume your provider quota.

College history requests are restricted to ordinary competitive-season windows:
NCAAB November–March plus April 1–15, and NCAAF August–January. Exhibition,
spring-football, and exceptional off-calendar games are not included in this
baseline's historical collection. Selected-date schedules are still requested
explicitly. If ESPN rejects a multi-day request with HTTP 400, the collector
retries individual dates and caches the range only after every daily request
succeeds. A daily error is surfaced, never silently treated as an empty schedule.
Daily fallback can increase first-run download time; subsequent runs reuse caches.

## College team-strength research: weekly walk-forward v2

Select NCAAB or NCAAF, open **Model performance**, and click **Run [sport] weekly
research comparison**. Leave **Reuse downloaded college research history** checked
to use the same input as the previous report without downloading again. If cached
research history is unavailable, or reuse is unchecked, the app downloads three
completed seasons plus the current season. No paid odds requests are made.

Before each test week, both models receive exactly the same completed games from
the prior 730 days, available before Monday 00:00 America/Chicago minus one
microsecond. Ratings stay frozen for the entire test week. The Elo algorithm is
replayed on that history; the candidate refits opponent-adjusted margin ratings.
This weekly Elo policy differs from the live game-by-game model and from the
older seasonal research comparison. Compare the models within this report;
do not treat changed scores versus the old report as proof of improvement.

Fixed candidate settings: Ridge alpha=10, 365-day recency half-life, scoring
margins capped at 40 basketball points or 45 football points, and a learned home
advantage (zero on neutral sites). A sigmoid probability mapping, C=1, trains only
on earlier weekly out-of-sample margins whose results were available before the
current cutoff. Calibration uses the prior 365 days, at least 50 games, both
outcome classes, and established-team games. Historical margins are not recomputed
using the latest model. The first season supplies training, the second generates
calibration predictions, and evaluation starts in the third season. Weeks lacking
sufficient data are explicitly reported as skipped.

Both models are assessed on identical games. Coverage tables separate established
history (at least 10 prior games for each participant), limited history (1–9), and
unseen participants. Unknown participants use zero candidate team coefficients;
the overall results still include them. Missing-history games must not be
interpreted as trustworthy forecasts. Reliability tables are split by coverage
and season. Inspect counts before comparing probability bins.

Exports: college seasons, weeks (training/calibration cutoff audit), predictions,
coverage, calibration_predictions (out-of-sample audit), and reliability_by_season.
Reports use `artifacts/sports_lab/[sport]_team_strength_weekly_v2`, preserving old
seasonal report files. Inputs remain in
`data/sports_lab/college/[sport]_research_history.csv`; hashes and latest-result
timestamps identify the input used. The research run never overwrites paper picks
or changes live models, probabilities, or model versions.

The candidate is still a scoring-margin model, not possession-adjusted efficiency
or EPA. It lacks transfers, current starters, injuries, and coaching context.
Historical publication times are conservatively reconstructed; source corrections
can still affect the data. Current-season samples are partial. Reviewed test
periods become development data, and future games must confirm a decision to
promote a model. No profitable-betting claim or superiority to sportsbook prices
is inferred without historical timestamped odds. No candidate is automatically
promoted, regardless of the research scores.

## Prospective paired college testing

After reviewing the weekly report, choose NCAAB or NCAAF under Find value and
select an upcoming game date. Click **Record both models for upcoming [sport]
games**. The app refreshes recent public results, merges them with the cached
extended research history (normalizing IDs to avoid duplicate counting), and saves
both weekly Elo and margin-candidate home-win probabilities. No odds download,
paid request, or paper-pick selection is required. Complete the research-history
download for that sport first if the app requests it.

Both models use the weekly v2 protocol. The training cutoff is the beginning of
the target week, or the generation time when forecasting before that week begins.
Calibration uses only available earlier out-of-sample predictions. Model versions,
cutoffs, and generation times are recorded. Only games with at least 10 available
prior games per team are saved; excluded counts are shown. A game must still be
upcoming when the record is written. The first paired snapshot per sport/event
is immutable, even if the button is clicked again after refreshed data arrives.

See **Model performance → Paired prospective forecasts** to confirm pending
records. After games finish, use **Results tracker → Refresh college finals from
public data**. The opt-in 15-minute automatic college refresh also uses public
finals while the app session is open. Public event IDs allow grading independently
of sportsbook name matching. Both models are evaluated on the same decisive games;
voids are retained and excluded from accuracy/Brier/log-loss summaries. This is
forecast evaluation, not a betting ROI record, and it never changes existing picks
or promotes a model. Export paired forecasts for review as the sample grows.

Paired snapshots use a separate `shadow_forecasts` table in the existing
`data/sports_lab/paper.sqlite`. Preserve this database when updating code. As with
all college models here, roster/injury context and historical publication-time
limitations remain; probability estimates are experimental.

## College schedule and identity review

Public college downloads retain provider team IDs when available. Before new
college candidates or paired snapshots are accepted, the app checks whether the
same participant appears in different fixtures less than four hours apart, whether
a displayed name corresponds to conflicting IDs in that window, and whether the
same event ID has inconsistent participants/start times. The four-hour threshold
is a review heuristic, not proof of an invalid game; separate squads or provider
corrections may explain a flag. Name normalization is used when old saved records
lack provider IDs. It does not silently guess a replacement identity.

Flagged games are blocked from new paired records. Repeated recordings are also
checked against existing snapshots, so recording one conflicting game in a later
batch does not evade the check. Existing snapshots, winners, probabilities, and
paper bets are never deleted or rewritten. Schedule observations are retained in
an additive `shadow_quality_flags` table. Paired summaries exclude unresolved
review flags by default, and exports retain all original rows with review columns.
This changes the displayed evaluation cohort transparently, not the predictions.

Under **Model performance → Paired prospective forecasts**, use **Audit saved
[sport] schedules against public source** to refresh ESPN events for the saved
dates and inspect team IDs, start times, and final scores. Export the source
schedule audit for investigation. This reuses the same provider and is not an
independent verification of that provider's data. Upstream repeated fixtures must
be investigated before conclusions are drawn. Flags are not automatically cleared
or turned into void results. No sportsbook settlement rule is changed.

These checks cover selected public schedules and saved paired forecasts. They do
not retroactively certify or clean the entire historical training dataset. A
source audit can reveal upstream identity problems that require a separate data
correction before trusting model comparisons. Review flags depend only on schedule
and participant information, never on which model won or lost.

## Multi-sport collection dashboard

Open **Data collection** to see adapter availability and observed feed status for
NBA, NHL, NFL, NCAAB, NCAAF, and Tennis. Public schedule/results adapters are
available for the five team sports. Public tennis schedules are not yet connected;
select an active tournament for odds snapshots. Adapter availability is not proof
of a successful live request; inspect the observed status after collecting.

Select sports and a public schedule date, leave **Collect public schedules and
results** enabled, and click **Collect selected feeds now**. Odds archiving and
periodic collection are OFF by default. To archive prices, enable **Archive current
sportsbook odds** with ODDS_API_KEY configured. This makes one US-region two-way
moneyline request per selected sport or tennis tournament. Odds responses include
returned upcoming dates, not only the public schedule date. Provider credits vary;
the displayed request estimate is not a monthly cost guarantee.

For recurring collection, select a 15/30/60/120-minute interval and explicitly
enable **Collect periodically while this app session is open**. The same interval
is a per-feed cooldown for manual collection, including failed attempts. A recent
manual price snapshot also satisfies the cooldown. No collection runs while the
app session is closed, and this does not install a Windows background task. Reopen
and enable periodic collection again when needed. No requests run merely because
you open the tab. Fix and test odds access through Find value if required.

Snapshots are append-only JSON under
`data/sports_lab/collections/snapshots/[sport]/[source]`, with retrieval timestamps,
content hashes, row counts, and run status in `runs.sqlite`. Preserve the whole
collections directory when transferring computers. Failed requests do not replace
the last successful snapshot. Empty responses are recorded as empty, not fabricated
matches. Historical source observations are not backdated; collection starts now
and cannot reconstruct missing pregame prices from previous months. Event markers
indicate whether retrieval preceded the listed scheduled start, not verified live
market status. Requests and credentials are not archived as URLs or values.

Existing Find value odds fetches now archive their normalized events/prices without
an extra provider request. Automatic NFL metric generation also archives generated
inputs and source hashes; historical NFL inputs are reconstructed, not original
pregame archives. The collector does not itself retrain models, record bets, or
settle picks. Continue the NCAAB paired test and its public final refresh as before.

Advanced-statistics feeds remain disconnected for NBA, NHL, NCAAB, NCAAF, and
Tennis; player availability is not connected to the multi-sport models. The
coverage table makes these gaps explicit. A successful schedule or price download
does not mean the corresponding prediction model is validated or ready. Public
adapters depend on provider availability and may change. Cloud networking does not
currently allow live ESPN checks; NBA/NHL collection was tested with fixtures and
must be verified through the status panel on Windows.
