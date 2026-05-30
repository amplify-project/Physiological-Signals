#!/usr/bin/env pwsh
# Launch disengagement action download in external SSH window

$remoteHost = "eoghan@192.168.200.206"

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "📥 Launching Disengagement Actions Download" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""
Write-Host "Server: " -NoNewline -ForegroundColor Yellow
Write-Host $remoteHost
Write-Host "Actions: " -NoNewline -ForegroundColor Yellow
Write-Host "45 disengagement classes"
Write-Host ""
Write-Host "Opening external PowerShell window..." -ForegroundColor Cyan
Write-Host ""

# Simple SSH command that will work in external window
$command = "ssh $remoteHost `"cd ~/concert_engagement && python3 scripts/download_disengagement_actions.py`""

# Launch external window
Start-Process pwsh -ArgumentList @(
    "-NoExit",
    "-Command",
    $command
)

Write-Host "✓ External window launched!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Monitor progress in the external window" -ForegroundColor Cyan
Write-Host "📝 Download will show live progress for all 45 actions" -ForegroundColor Gray
Write-Host ""
