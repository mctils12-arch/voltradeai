@echo off
title VolTrade
color 0B

:: ── Get the folder this script lives in ──
set "APP_DIR=%~dp0"
if "%APP_DIR:~-1%"=="\" set "APP_DIR=%APP_DIR:~0,-1%"

cd /d "%APP_DIR%"

echo.
echo  VolTrade is starting...
echo  Your browser will open automatically.
echo  Keep this window open while using the app.
echo.

:: Open browser after a short delay (gives server time to start)
start "" cmd /c "timeout /t 5 /nobreak >nul && start http://localhost:5000"

:: Start the app
call npm run dev

:: If it exits, show message
echo.
echo  VolTrade has stopped. Press any key to close.
pause >nul
