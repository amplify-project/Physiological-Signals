@echo off
title Setup - Concert Engagement AR
echo ============================================================
echo  Concert Engagement AR System - First-time Setup
echo ============================================================
echo.

cd /d "%~dp0"

REM ---------------------------------------------------------------
REM Find a compatible Python (3.10, 3.11, or 3.12) via py launcher.
REM Python 3.13+ is BLOCKED - mediapipe holistic API was removed.
REM ---------------------------------------------------------------
set "PYEXE="

py -3.12 --version >nul 2>&1
if not errorlevel 1 set "PYEXE=py -3.12"
if defined PYEXE goto :found_python

py -3.11 --version >nul 2>&1
if not errorlevel 1 set "PYEXE=py -3.11"
if defined PYEXE goto :found_python

py -3.10 --version >nul 2>&1
if not errorlevel 1 set "PYEXE=py -3.10"
if defined PYEXE goto :found_python

echo.
echo ERROR: No compatible Python found.
echo.
echo This system requires Python 3.10, 3.11, or 3.12.
echo Python 3.13+ is NOT supported (mediapipe incompatible).
echo.
echo Install Python 3.12 from:
echo   https://www.python.org/downloads/release/python-3128/
echo During install tick "Add Python to PATH" and "tcl/tk and IDLE"
pause
exit /b 1

:found_python
for /f "tokens=*" %%v in ('%PYEXE% --version 2^>^&1') do echo Using: %%v
echo.

REM Wipe any previous broken venv for a clean install
if exist .venv (
    echo Found existing .venv - removing for a clean install...
    rmdir /s /q .venv
)

echo [1/6] Creating virtual environment...
%PYEXE% -m venv .venv
if errorlevel 1 (
    echo.
    echo ERROR: Could not create the virtual environment.
    echo Make sure Python 3.10, 3.11, or 3.12 is installed correctly.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

echo.
echo [2/6] Upgrading pip, setuptools and wheel inside the venv...
echo       (this prevents the red "new release of pip available" warning
echo        from appearing during the package installs below - it is cosmetic
echo        but looks alarming. Upgrading first makes the rest of setup quiet.)
python -m pip install --upgrade pip setuptools wheel --disable-pip-version-check
if errorlevel 1 (
    echo.
    echo ERROR: Could not upgrade pip. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo [3/6] Installing PyTorch 2.6.0 (CUDA 12.4)...
echo       (must be installed first from the PyTorch index)
python -m pip install --disable-pip-version-check torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

echo.
echo [4/6] Installing remaining dependencies...
echo       (constraints.txt locks torch so pip cannot downgrade it)
python -m pip install --disable-pip-version-check -r requirements.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu124

echo.
echo [5/6] Installing facenet-pytorch (face ID model)...
echo       (installed separately: its metadata has an outdated torch constraint)
echo       (the code is runtime-compatible with torch 2.6.0)
python -m pip install --disable-pip-version-check facenet-pytorch==2.6.0 --no-deps

echo.
echo [6/6] Ensuring face ID weights are available locally...
if not exist model mkdir model
if exist model\20180402-114759-vggface2.pt (
    echo       Face ID weights already present.
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -Uri 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt' -OutFile 'model\\20180402-114759-vggface2.pt' } catch { Write-Error $_; exit 1 }"
    if errorlevel 1 (
        echo.
        echo ERROR: Could not download the Face ID weights.
        echo.
        echo Download this file manually and place it here:
        echo   model\20180402-114759-vggface2.pt
        echo URL:
        echo   https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt
        pause
        exit /b 1
    )
)

echo.
echo [6b/6] Ensuring long-range face models (YuNet + SFace) are available...
if exist model\face_detection_yunet_2023mar.onnx (
    echo       YuNet detector already present.
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -Uri 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx' -OutFile 'model\\face_detection_yunet_2023mar.onnx' } catch { Write-Error $_; exit 1 }"
    if errorlevel 1 (
        echo WARNING: Could not download YuNet. Long-range face ID will fall
        echo          back to MTCNN. Manual URL:
        echo   https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
    )
)
if exist model\face_recognition_sface_2021dec.onnx (
    echo       SFace recognizer already present.
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -Uri 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx' -OutFile 'model\\face_recognition_sface_2021dec.onnx' } catch { Write-Error $_; exit 1 }"
    if errorlevel 1 (
        echo WARNING: Could not download SFace. Long-range face ID will fall
        echo          back to MTCNN. Manual URL:
        echo   https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
    )
)

echo.
echo ============================================================
echo  Verifying key package versions...
echo ============================================================
python -m pip list --disable-pip-version-check | findstr /i "torch torchvision torchaudio facenet ultralytics mediapipe pillow numpy opencv"

echo.
echo ============================================================
echo  SETUP COMPLETE
echo ============================================================
echo.
echo NOTE: If you see "facenet-pytorch has requirement torch less than 2.3.0"
echo       this is a known false alarm. The code is fully compatible with
echo       torch 2.6.0. Ignore it.
echo.
echo Done!
echo.
echo  To run the system:
echo    1_START_REDIS.bat       - start Redis (keep this window open)
echo    2_START_EMOTIBIT.bat    - start EmotiBit physiological publisher
echo    3_START_ENGAGEMENT.bat  - start camera + AI engagement inference
echo.
pause
