@echo off
setlocal enabledelayedexpansion
title VolTrade — Installer
color 0B
echo.
echo  ========================================
echo    VolTrade Installer
echo  ========================================
echo.

:: ── Get the folder this script lives in ──
set "APP_DIR=%~dp0"
:: Remove trailing backslash
if "%APP_DIR:~-1%"=="\" set "APP_DIR=%APP_DIR:~0,-1%"

:: ── Check for Node.js ──
echo  [1/4] Checking for Node.js...
where node >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo  ERROR: Node.js is NOT installed.
    echo.
    echo  Please install it from:
    echo    https://nodejs.org
    echo.
    echo  Download the LTS version, install it, then run this script again.
    echo.
    pause
    exit /b 1
)
for /f "delims=" %%v in ('node --version') do set NODE_VER=%%v
echo  Found Node.js %NODE_VER%

:: ── Check for Python ──
echo  [2/4] Checking for Python...
where python >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo  ERROR: Python is NOT installed.
    echo.
    echo  Please install it from:
    echo    https://www.python.org/downloads/
    echo.
    echo  IMPORTANT: Check the box "Add Python to PATH" during install.
    echo  Then run this script again.
    echo.
    pause
    exit /b 1
)
for /f "delims=" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  Found %PY_VER%

:: ── Install Python dependencies ──
echo  [3/4] Installing Python packages (yfinance, scipy)...
python -m pip install --quiet --upgrade yfinance scipy numpy 2>nul
if %errorlevel% neq 0 (
    echo  Warning: pip had an issue. Trying with --user flag...
    python -m pip install --quiet --upgrade --user yfinance scipy numpy 2>nul
)
echo  Python packages ready.

:: ── Install Node dependencies ──
echo  [4/4] Installing Node.js packages (this may take a minute)...
cd /d "%APP_DIR%"
call npm install 2>nul
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo  ERROR: npm install failed.
    echo  Please make sure you have internet access and try again.
    echo.
    pause
    exit /b 1
)
echo  Node packages ready.

:: ── Create Desktop shortcut ──
echo.
echo  Creating Desktop shortcut...

set "DESKTOP=%USERPROFILE%\Desktop"
set "SHORTCUT=%DESKTOP%\VolTrade.lnk"
set "BAT_PATH=%APP_DIR%\VolTrade.bat"

:: Use PowerShell to create a proper .lnk shortcut with the VolTrade icon
set "ICO_PATH=%APP_DIR%\VolTrade.ico"
powershell -NoProfile -Command ^
  "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%BAT_PATH%'; $s.WorkingDirectory = '%APP_DIR%'; $s.IconLocation = '%ICO_PATH%'; $s.Description = 'Launch VolTrade'; $s.Save()"

if exist "%SHORTCUT%" (
    echo  Shortcut created on your Desktop!
) else (
    echo  Could not create shortcut automatically.
    echo  You can manually create one for: %BAT_PATH%
)

:: ── Done ──
echo.
color 0A
echo  ========================================
echo    Installation complete!
echo  ========================================
echo.
echo  To start VolTrade:
echo    Double-click "VolTrade" on your Desktop
echo.
echo  Or double-click "VolTrade.bat" in this folder.
echo.
pause
