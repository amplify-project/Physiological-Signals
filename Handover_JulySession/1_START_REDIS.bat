@echo off
title Redis Server — Concert Engagement AR
echo Starting Redis server...
echo Press Ctrl+C in this window to stop Redis.
echo.
"%~dp0redis\redis-server.exe" "%~dp0redis\redis.windows.conf"
pause
