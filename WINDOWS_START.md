# Windows setup

Download this repository from GitHub using **Code → Download ZIP**, then right-click
the downloaded ZIP and choose **Extract All**. Open the extracted project folder.
It should contain `pyproject.toml`, `requirements.lock`, and `src`.

Click the File Explorer address bar, type `powershell`, and press Enter. This opens
PowerShell in the correct folder. Run each command below separately, pressing Enter
after each command. You do not need an Administrator window.

## Install

Use **64-bit Python 3.12** for this project. Python 3.15 is not supported by the
pinned PyArrow dependency. Install 3.12 alongside your existing Python:

```powershell
winget install --exact --id Python.Python.3.12
```

Close and reopen PowerShell in the project folder, then check:

```powershell
py -3.12 --version
```

If you already created `.venv` with Python 3.15, preserve it before creating the new
environment (run this rename only once):

```powershell
Rename-Item .venv .venv-python315
```

```powershell
py -3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -c requirements.lock -e ".[dev]"
```

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The cloud version was tested with Python 3.12. If installation fails, share the error
output, without any credentials. The Windows commands have not been executed on a
Windows machine during this development session.

## Prepare the historical models

This downloads about 200 MB of public NFL data. GitHub contains source code only;
prepared models, historical data, credentials, and virtual environments are excluded.

```powershell
.\.venv\Scripts\nfl-model.exe --feature-set enhanced --end-season 2024 --holdout-season 2024 --development-only --output-dir artifacts/enhanced-development
```

```powershell
.\.venv\Scripts\nfl-calibrate.exe --schedule data\schedules\reference_games.csv
```

Optional closing-line diagnostic:

```powershell
.\.venv\Scripts\nfl-backtest.exe --closing-reference --schedule data\schedules\reference_games.csv --output-dir artifacts/closing-reference
```

The current selection rule lost approximately 8% in this diagnostic. These reference
prices have no observation timestamps and do not establish an implementable or
profitable strategy.

## Generate a forecast

For the matchup discussed during development:

```powershell
.\.venv\Scripts\nfl-update.exe --date 2026-10-12 --away BUF --home LA --refresh
```

For another game, substitute its date and team abbreviations. The date must appear
in the NFL schedule. Historical football data and source context can run without an
odds key; live sportsbook prices require your provider key. Configure `ODDS_API_KEY`
in your computer's environment-variable settings and open a new PowerShell window
before running the updater. Cloud environment secrets do not transfer to your PC.
Never share your key in chat or add it to the repository.

## Launch the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\nfl_model\dashboard.py
```

Open the **Local URL** that Streamlit prints in your browser. Choose the same game
date and matchup as your forecast. Keep the PowerShell window open while using the
app; press Ctrl+C there to stop it.

The dashboard displays expected starters, injuries, weather, probability validation,
prices when available, and paper bets. Weather and most injuries are contextual
information, not learned score adjustments. All forecasts are research estimates;
the application does not place real wagers.
