@echo off
setlocal
cd /d "%~dp0"
if not exist "requirements.lock" (
  echo Missing project files. Extract the entire ZIP before running this launcher.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -m pip install -c requirements.lock -e ".[dev]"
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m streamlit run src/sports_lab/dashboard.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
if errorlevel 1 goto :failed
exit /b 0
:failed
 echo Startup failed. Read the error above. Python 3.12 is required for initial setup.
 pause
 exit /b 1
