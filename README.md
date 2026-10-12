# NFL Prediction Model

A first research pipeline for NFL pregame scoring predictions. It downloads public
nflverse play-by-play, builds prior-game features, trains Ridge linear regression
and Poisson regression, and evaluates them chronologically. An enhanced feature set
adds opponent-adjusted ratings and quarterback-history proxies. The research dashboard
compares timestamped prices and can log paper candidates; it does not place real bets.

For Windows installation, start with [WINDOWS_START.md](WINDOWS_START.md).
GitHub contains source only; datasets and fitted models are generated locally.

## Run

Python 3.11–3.14 is supported; use Python 3.12, the tested version. Python 3.15
cannot install the pinned PyArrow wheel. Windows setup explicitly selects 3.12.

```bash
cd /workspace/SportsPredicitonModel
python3 -m venv .venv
.venv/bin/python -m pip install -c requirements.lock -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/nfl-model
```

The default run downloads 2016–2025 regular-season play-by-play, trains initially
on 2016–2020, and evaluates 2021–2025. Each season's models use only previous
seasons. The last evaluation season is designated the final holdout. Once its
results influence development, it is no longer an untouched test set.

Raw downloads are about 20 MB per season, cached under `data/raw/`. Later runs
reuse the cache. Data and fitted models are intentionally ignored by Git.
HTTPS certificate verification remains enabled. Historical football data needs no
credentials; live/historical odds require a securely configured provider key.

For a shorter development run that does not inspect the final holdout:

```bash
.venv/bin/nfl-model --start-season 2019 --end-season 2023 \
  --first-test-season 2022 --holdout-season 2023 \
  --output-dir artifacts/development
```

## Features and targets

Each row represents one team in one game. Targets are final team points; opposing
team predictions are paired to calculate expected margin and total. Both models
predict team points rather than directly optimizing margin or total.

For both teams, exponentially weighted prior-game averages include:

- Overall, passing, rushing, and early-down EPA.
- Positive-EPA success rate.
- Explosive-play rate: passes gaining 20+ yards or runs gaining 10+ yards.
- Sacks per pass attempt, including sack plays.
- Offensive and defensive scrimmage-play counts.

Home field, prior-game count, and rest days (capped at 21) are also included.
Kneel-downs and spikes are excluded. Features are shifted before averaging, so
current-game outcomes cannot enter current-game inputs. The first observed game
has missing historical features, imputed using training data only. Rolling
features carry across seasons. Franchise aliases such as OAK/LV and SD/LAC are
normalized before aggregation. These are all-game averages, not neutral-situation statistics.

The feature construction is in `src/nfl_model/features.py`; model fitting and
reporting are in `src/nfl_model/evaluation.py`. Imputation and scaling are fitted
inside each training fold. Hyperparameters are fixed defaults, with no holdout tuning.

## Enhanced features and fair comparison

Opponent-adjusted offense and defense ratings are fitted to overall, passing,
and rushing EPA using a Ridge model with separate offense-team and defense-team
effects and a home-field indicator. Positive defensive effects mean more EPA
allowed. Ratings express deviations from the league baseline, not raw EPA.
Each NFL week's ratings use only games completed before the first game of that
week, with up to two years of history, a 365-day recency half-life, play-count
weights, and fixed regularization. They do not use final season ratings.

The quarterback feature uses the team's **previous game's primary passer**, identified
by the most recorded passing dropbacks with a player ID. It looks up that player's
prior rolling passing EPA, accumulated dropbacks, and time since the last appearance.
All games on a date receive features before that date's results update the history.
This is a continuity proxy, not the expected starter: injuries, offseason moves,
and new starters require separate point-in-time availability data. Passing EPA also
reflects teammates and opponents, not quarterback ability alone. The most recent
primary passer's history follows the player across teams when already observed.

Reproduce the development comparison without reading 2025:

```bash
.venv/bin/nfl-model --feature-set baseline --end-season 2024 \
  --holdout-season 2024 --development-only --output-dir artifacts/baseline-development
.venv/bin/nfl-model --feature-set enhanced --end-season 2024 \
  --holdout-season 2024 --development-only --output-dir artifacts/enhanced-development
.venv/bin/nfl-compare
```

Here `--holdout-season` selects the last evaluated season; `--development-only`
marks every season as development, with no claim of an untouched holdout.
Both versions use identical source files, training years, outcomes, and fixed
model hyperparameters. `nfl-compare` verifies those source hashes and prediction
alignment and rejects holdout-labeled reports. Outputs are saved separately,
including `artifacts/feature_comparison.md` and `.csv`.

The original 2025 baseline results have already been inspected. Use development
comparisons for subsequent feature decisions, then reserve a new unseen period
for a final test. Default `nfl-model` reproduces the initial baseline experiment;
it is not the recommended command for iterative feature tuning.

## Outputs

- `data/pregame_features_<feature-set>_<start>_<end>.parquet`: targets, source metrics, and pregame features.
  The original baseline dataset remains at `data/pregame_features.parquet`.
- `artifacts/report.md`: season-by-season evaluation and limitations.
- `artifacts/metrics.csv`: score, margin, total errors and winner accuracy.
- `artifacts/predictions.csv`: individual historical team predictions.
- `artifacts/feature_names.json`: the exact model input columns.
- `artifacts/data_manifest.json`: source cache paths and SHA-256 digests for this run.
- `artifacts/run_config.json`: feature set, season range, and evaluation mode.
- `artifacts/{ridge,poisson}_<season>.joblib`: evaluation models trained before that season.
- `artifacts/{ridge,poisson}_latest.joblib`: research models fitted through the holdout season.

Only load joblib models you trust; pickle-based model files can execute code.
Latest models are not used to produce historical evaluation predictions. Upcoming
predictions are available through `nfl-predict`, described below.

## Upcoming-game forecasts

```bash
.venv/bin/nfl-predict --date 2026-10-12 --away BUF --home LA
```

The command verifies the matchup against the public nflverse schedule, then loads
play-by-play through the requested season. Raw data is filtered to dates strictly
before the matchup **before aggregation**, even when cached files contain later
games. Training uses seasons before the matchup season; current-season completed
games update pregame features but do not refit the scoring models. Target-game
rows contain no outcome or current-game starter data.

Outputs under `artifacts/upcoming/<game_id>/` include `prediction.json`,
`pregame_inputs.csv`, fitted Ridge/Poisson models, and source-file hashes. The JSON
records the generation time, training cutoff, latest observed game date, and
limitations. Schedule snapshots are cached under `data/schedules/` by prediction
date; raw season files are also cached. For a later refresh, preserve the old
snapshot/run, then explicitly refresh the relevant cached schedule and current
season file. The command does not automatically guarantee fresh data.

These are expected points, not exact-score probabilities. The standalone
`nfl-predict` command provides points; `nfl-update` adds the separately fitted
probability calibration and price comparison. No profitable-betting claim is provided.
The prior-game quarterback proxy does not handle a newly announced starter or
injury. Schedule columns containing scores, QB names, weather, or betting lines
are never passed to the models. Only matchup identity, season, week, and date
are read from the selected schedule row.

## Validation

Tests exercise offense/defense aggregation, spike exclusion, opponent pairing,
current/future-game feature leakage, chronological training, and positive Poisson
predictions. Enhanced tests also exercise quarterback changes, frozen weekly ratings,
relative team-strength recovery, franchise aliases, and comparison input integrity.
Upcoming-game tests verify that target/future games cannot change a forecast and
that current-season score labels are not fitted. `pytest` uses synthetic fixtures without network access; the full
CLI run validates the real data workflow. Repeating it should reproduce metrics
with the cached data and the same dependency versions.

## Next steps

1. Evaluate opponent and quarterback features separately to understand their individual contributions.
2. Add expected starting quarterback, timestamped injuries, and forecast weather.
3. Establish stronger team-rating and market benchmarks, plus calibrated probabilities.
4. Obtain timestamped historical odds and prices; closing odds are not valid substitutes for earlier available prices.
5. Backtest value decisions with pushes, vig, and uncertainty, then paper bet.
6. Add a dashboard, scheduled source refresh, and a prospective paper-betting log.

Poisson point counts do not capture NFL's scoring increments, overdispersion, or
correlated team scores. Test these assumptions before deriving betting probabilities.
Live odds remain dependent on a configured provider credential. Probability
calibration and a negative closing-reference diagnostic are documented below;
no actionable historical return or profitability is claimed.
Historical source files may include later corrections; they are not point-in-time archives.

## Data source

Play-by-play comes from [nflverse-data](https://github.com/nflverse/nflverse-data),
using its `pbp` release assets. Consult the upstream repository's data license,
attribution, and usage terms before redistributing data. Generated raw files are
not committed here. Download hashes document the local inputs; they are not
independently published upstream checksums.

## Probability calibration and betting evaluation

```bash
.venv/bin/nfl-calibrate
.venv/bin/nfl-backtest --closing-reference --output-dir artifacts/closing-reference
```

Calibration uses 2021 out-of-sample score residuals, initially fits on 2022,
and evaluates 2023–2024 with only earlier calibration observations. The latest
bundle fits calibration through 2024. Ridge resamples paired team-score errors;
Poisson builds independent score-count distributions. Final tie mass is adjusted
to the residual season's observed NFL tie rate. Market-specific logistic calibration
estimates conditional win probability; base push probability is retained.
Opposing sides are calibrated as complements. Score distributions remain research
approximations, particularly for NFL key margins and overtime.

The evaluation writes `artifacts/probability/{report.md,metrics.csv,reliability.csv,
validation.csv,calibration.joblib}`. Brier/log loss exclude actual pushes; they are
undefined for groups with no decisive games. Calibration did not improve every
season/market. These are development results, not final prospective validation.

The closing-reference diagnostic selected 502 flat-unit bets in 2023–2024 and
lost about 40.20 units (-8.01% ROI). Its prices have no observation timestamps,
so it cannot establish what was available an hour before kickoff or prove an
implementable return. It does not support a profitable-strategy claim.

A timestamped backtest requires authentic historical quotes:

```bash
.venv/bin/nfl-backtest --odds-csv /path/to/historical_quotes.csv \
  --decision-minutes 60 --output-dir artifacts/timestamped-backtest
```

Quotes must be captured before kickoff, contain explicit timezone offsets, and
be fresh within 15 minutes of the decision cutoff. The latest line per book,
market, and side is used; a replaced, more favorable line is not retained as an
available price. Calibrators are refitted using earlier seasons only. The strategy
chooses at most one bet per game, uses one flat unit, requires 3% estimated return
and positive estimated return from both models, and includes losses and pushes.
Reports include ROI and drawdown. This fixed selection rule is experimental.
Timestamped injuries/weather and a new untouched period are still needed for
stronger validation; importing quote timestamps does not fix retrospective
corrections in football source data.

## Live data, snapshots, and updates

Store `ODDS_API_KEY` securely in cloud environment settings, not in a file or chat.
Required allowed domains are `site.api.espn.com`, `api.open-meteo.com`, and
`api.the-odds-api.com`, in addition to existing package/GitHub presets.
Restart the dashboard/updater after changes to injected credentials.

```bash
.venv/bin/nfl-update --date 2026-10-12 --away BUF --home LA --refresh
.venv/bin/nfl-odds --date 2026-10-12 --away BUF --home LA
```

The updater fetches ESPN depth charts and roster injury statuses, kickoff weather
from Open-Meteo, current NFL play-by-play, and sportsbook prices. Depth charts are
expected-starter indications, not official game-day confirmation. ESPN player IDs
are mapped to nflverse IDs using the roster snapshot. Current SoFi weather can be
fetched via a coordinate fallback; other venues need provider coordinates.

Timestamped starter overrides use that player's previous primary-passer statistics
in the existing quarterback feature coefficients. Overrides are not separately
validated. Other injury statuses and forecast weather are displayed context,
**not fitted score adjustments**. The public 2025/2026 injury files checked during
implementation have no publication timestamps; using them as known pregame
announcements would not be a credible point-in-time backtest.

Each refresh preserves a separate run under `artifacts/upcoming/<game_id>/runs/`
with forecast, model inputs, fitted models, source snapshots, source hashes,
prices, and estimated value. `artifacts/dashboard/state.json` points to the latest
run. Refreshing current-season play-by-play archives the prior file by content
hash. Provider request URLs containing authentication are never logged or saved.
Failed providers produce explicit source-status messages, not fabricated data.

Continuous updates are supported:

```bash
.venv/bin/nfl-update --date 2026-10-12 --away BUF --home LA --refresh \
  --watch-seconds 3600
```

Choose frequency according to the provider quota; a query requests three markets
in the US region. Shorten the interval near kickoff if quota permits. Quotes older
than 15 minutes cannot qualify, even between scheduled updates. Live processes
must restart after environment restoration. No unattended watcher is required to
view an existing forecast.

Fetch one historical snapshot when the subscription permits:

```bash
.venv/bin/nfl-odds --date 2024-12-08 --away BUF --home LA \
  --historical-at 2024-12-08T20:25:00Z
```

Historical access may require a paid plan and consumes provider credits. Bulk
historical downloads are not automatically started. The client rejects a snapshot
later than the requested cutoff and only maps events matching the game date.

## CSV and context imports

`examples/odds_template.csv` contains the required header. Markets are `moneyline`,
`spread`, and `total`; sides are `home`/`away` or `over`/`under`. `point` is the
selected side's handicap for spreads and the scoring threshold for totals.
`observed_at`, `source_updated_at`, and `commence_time` require explicit timezones.
Use authentic source timestamps; do not assign today's retrieval time to old lines.
American odds must have absolute magnitude at least 100.

Reviewed context JSON may contain `source`, `observed_at`, `commence_time`,
`lineups_reviewed`, `teams`, `weather`, and `venue`. Each team may supply
`expected_qb_gsis_id`, `expected_qb` (name and source basis), and `injuries`.
Only mark `lineups_reviewed` true after reviewing the actual report for this game.
Timestamps must precede kickoff. Forecasts reject future context and mismatched
game dates. Unknown backup quarterbacks cannot receive invented ratings.

```bash
.venv/bin/nfl-update --date 2026-10-12 --away BUF --home LA \
  --context-json /path/to/reviewed_context.json --odds-csv /path/to/current_quotes.csv
```

The odds provider and CSV importer can calculate price-specific expected return
and normalize paired prices to remove vig. Only paired opposing outcomes from the
same book, line, and snapshot receive a no-vig comparison. Estimated value is not
an established edge; no real wagers are placed.

## Dashboard and paper ledger

```bash
.venv/bin/streamlit run src/nfl_model/dashboard.py --server.address 127.0.0.1 \
  --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

The dashboard includes forecast, odds/value, probability validation, paper bets,
and source-status tabs. It displays the negative closing-line diagnostic alongside
current estimates. The sidebar supports source refresh and CSV/context imports.
Reviewing the listed starters/injuries/weather enables paper-candidate checks;
missing starters, stale context/quotes, started games, unavailable calibration,
small estimated edges, or disagreement between models block candidates.

`data/paper.sqlite` stores quote snapshots and immutable paper bets, including the
original price, forecast JSON, and forecast hash. The dashboard can log the top
eligible candidate; the updater's optional `--paper` flag logs at most one eligible
paper bet per game. Repeated updates cannot overwrite that bet. It settles from
scores on dates strictly before the current date, preserving pushes and payouts.
It tracks profit, ROI, and drawdown. No prospective paper-bet performance is
claimed before bets are recorded and games finish.

Dashboard readiness was checked through HTTP responses and Streamlit AppTest,
including rendered forecasts/prices and disabled betting controls for unreviewed
inputs. Provider and settlement fixtures are automated tests, not claims of
successful live odds access or completed prospective paper betting.
