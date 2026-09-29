@echo off
title EmotiBit Publisher — Concert Engagement AR
echo Starting EmotiBit physiological publisher (multi-device)...
echo Power on your EmotiBit device(s) before continuing.
echo Press ENTER in this window once devices are discovered.
echo.
echo If Redis runs on ANOTHER device, start this with the hub IP, e.g.:
echo    2_START_EMOTIBIT.bat --redis-host 192.168.1.50
echo.
cd /d "%~dp0"
call .venv\Scripts\activate.bat 2>nul || (
    echo ERROR: .venv not found. Run SETUP.bat first.
    pause
    exit /b 1
)
python multiemotibit_UDP_SD_RFv2.py %*
pause
