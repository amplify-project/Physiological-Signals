@echo off
title Engagement Inference — Concert Engagement AR
echo Starting engagement inference (camera + AI model)...
echo.
cd /d "%~dp0"
call .venv\Scripts\activate.bat 2>nul || (
    echo ERROR: .venv not found. Run SETUP.bat first.
    pause
    exit /b 1
)
python live_multiperson_binary_v2.py --select-camera --save %*
pause
