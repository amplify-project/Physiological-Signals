@echo off
setlocal
title Redis Service Advertiser - Concert Engagement AR
echo Starting Redis service advertiser...
echo The advertiser will wait until Redis is available on localhost:6379.
echo Press Ctrl+C in this window to stop advertising.
echo.
cd /d "%~dp0"
call .venv\Scripts\activate.bat 2>nul || (
    echo ERROR: .venv not found. Run SETUP.bat first.
    pause
    exit /b 1
)
python redis_service_advertiser.py %*
if errorlevel 1 (
    echo.
    echo Redis service advertiser exited with an error.
)
pause
