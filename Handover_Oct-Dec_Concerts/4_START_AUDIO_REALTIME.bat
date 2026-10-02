@echo off
title Audio time recording
echo Starting real-time audio detection (music ^> CSV + WAV)...
echo This runs saves audio timestamps to CSV alongside EmotiBit data.
echo.
cd /d "%~dp0"
call .venv\Scripts\activate.bat 2>nul || (
    echo ERROR: .venv not found. Run SETUP.bat first.
    pause
    exit /b 1
)

REM Check if required packages are installed
python -c "import sounddevice, soundfile, onnxruntime, torch" 2>nul || (
    echo ERROR: Audio packages not installed. Installing now...
    REM constraints.txt locks torch to the cu124 build so this fallback
    REM cannot downgrade the GPU torch to a CPU wheel.
    pip install sounddevice==0.5.1 soundfile==0.13.1 onnxruntime==1.20.1 -c constraints.txt
)

REM Device selection: interactive by default - the script lists detected
REM microphones and asks the user to pick one (Enter = system default).
REM Pass a device ID to skip the prompt:      4_START_AUDIO_REALTIME.bat 1
REM Pass "auto" for the loudest-mic RMS probe: 4_START_AUDIO_REALTIME.bat auto
REM (run "python audio_monitor_csv.py --list-devices" to see the IDs)
set ARGS=
if "%~1"=="" (
    echo You will be asked to choose a microphone from a list.
) else if /i "%~1"=="auto" (
    echo Auto-detecting microphone by sound level...
    set ARGS=--auto-device
) else (
    echo Using microphone device ID %~1
    set ARGS=--device %~1
)

echo.
echo Starting audio detection...
echo Output: emotibit_recordings\audio_YYYY-MM-DD_HH-MM-SS.csv + .wav
echo WAV auto-saves every 5 seconds (safe even if you forget to stop)
echo Stop with Ctrl+C or just close this window (both save properly!)
echo.

python audio_monitor_csv.py --quiet %ARGS%

pause
