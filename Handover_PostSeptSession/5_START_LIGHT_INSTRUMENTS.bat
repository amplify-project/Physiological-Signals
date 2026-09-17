@echo off
title Light Instruments Subscriber — Concert Engagement AR
echo Starting Light Instruments Redis listener...
echo This logs Tom's node-editor app's events to CSV alongside EmotiBit/audio data.
echo.
cd /d "%~dp0"
call .venv\Scripts\activate.bat 2>nul || (
    echo ERROR: .venv not found. Run SETUP.bat first.
    pause
    exit /b 1
)

REM Default assumes Tom's app publishes to THIS machine's Redis (localhost).
REM If Tom's app is running on his own laptop instead, pass its network
REM address:  5_START_LIGHT_INSTRUMENTS.bat --host 192.168.x.x
python redis_subscriber_light_instruments.py %*
pause
