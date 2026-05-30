#!/usr/bin/env pwsh
# Watch live training output (training continues even if you close this)

Write-Host "📺 Live Training Output" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop watching (training continues in background)" -ForegroundColor Yellow
Write-Host ""

ssh eoghan@192.168.200.206 'tail -f ~/concert_engagement/logs/training_kinetics_transformer_dataparallel.log'
