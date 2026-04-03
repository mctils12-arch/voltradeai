@echo off
cd /d "%~dp0"
echo.
echo Starting VolTrade...
echo Keep this window open while using the app.
echo.
if not exist node_modules (
    echo Installing packages - please wait 1-2 minutes...
    python -m pip install --quiet --upgrade yfinance scipy numpy
    call npm install
)
start "" http://localhost:5000
npx cross-env NODE_ENV=development tsx server/index.ts
