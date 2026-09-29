@echo off
setlocal
cd /d "%~dp0"
title Redis Server — Concert Engagement AR

REM Launch the mDNS advertiser alongside Redis so clients (EmotiBit publisher,
REM engagement GUI, AR glasses) auto-discover this hub without a hardcoded IP.
REM It waits for Redis to answer PING, then announces this PC's LAN IP:6379 and
REM re-announces if the address changes. Runs in its own window; close that
REM window to stop advertising. Extra args to this batch pass through to it
REM (e.g. --advertise-address 192.168.1.50 to force a specific interface).
if exist ".venv\Scripts\activate.bat" (
    echo Starting mDNS advertiser so clients can auto-discover this broker...
    start "Redis Advertiser" cmd /k "call .venv\Scripts\activate.bat && python redis_service_advertiser.py %*"
) else (
    echo WARNING: .venv not found — auto-discovery advertiser NOT started.
    echo          Run SETUP.bat first, or point clients with --redis-host ^<ip^>.
)
echo.

echo Starting Redis server...
echo Press Ctrl+C in this window to stop Redis.
echo (Close the "Redis Advertiser" window separately to stop advertising.)
echo.
"%~dp0redis\redis-server.exe" "%~dp0redis\redis.windows.conf"
pause
