@echo off
title Light Instruments Subscriber (Remote) — Concert Engagement AR
echo Starting Light Instruments Redis listener -- REMOTE mode.
echo Use this ONLY if Tom's laptop runs BOTH his app AND its own Redis server,
echo and this machine is just a subscriber over the network.
echo (If Redis runs on THIS machine instead, use 5_START_LIGHT_INSTRUMENTS.bat.)
echo.
cd /d "%~dp0"
call .venv\Scripts\activate.bat 2>nul || (
    echo ERROR: .venv not found. Run SETUP.bat first.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo ERROR: Tom's laptop IP address is required.
    echo Usage: 5_START_LIGHT_INSTRUMENTS_REMOTE.bat 192.168.1.23
    echo Find it on Tom's laptop by running "ipconfig" there.
    pause
    exit /b 1
)

python redis_subscriber_light_instruments_remote.py --host %1 %2 %3 %4 %5
pause
