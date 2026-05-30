#!/usr/bin/env pwsh
# Launch feature extraction in external window

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "🚀 Launching Feature Extraction" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""
Write-Host "Server: eoghan@192.168.200.206" -ForegroundColor Yellow
Write-Host "GPUs: 12x Tesla T4" -ForegroundColor Yellow
Write-Host "Workers: 12" -ForegroundColor Yellow
Write-Host ""

# Launch external window
$command = "ssh eoghan@192.168.200.206 'cd ~/concert_engagement && python3 scripts/extract_kinetics_features.py --workers 12'"

Start-Process pwsh -ArgumentList "-NoExit", "-Command", $command

Write-Host "✓ External window launched!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Now starting resource monitor in VS Code..." -ForegroundColor Cyan
Write-Host ""
