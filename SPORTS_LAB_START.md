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
