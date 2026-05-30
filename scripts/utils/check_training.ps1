#!/usr/bin/env pwsh
# Simple one-time training status check (more reliable than looping dashboard)

$server = "eoghan@192.168.200.206"

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Training Status Check - $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check if process is running
Write-Host "Checking process..." -ForegroundColor Yellow
$proc = ssh $server 'ps aux | grep "python.*train_action_transformer" | grep -v grep'
if ($proc) {
    $procId = $proc.Split()[1]
    $cpu = $proc.Split()[2]
    $mem = $proc.Split()[3]
    Write-Host "✅ Training RUNNING - PID: $procId | CPU: $cpu% | MEM: $mem%" -ForegroundColor Green
} else {
    Write-Host "❌ Training NOT RUNNING" -ForegroundColor Red
}
Write-Host ""

# Check log for latest output
Write-Host "Latest log output:" -ForegroundColor Yellow
ssh $server 'tail -20 ~/concert_engagement/logs/training_kinetics_transformer_ddp.log 2>/dev/null'
Write-Host ""

# Check GPU usage summary
Write-Host "GPU Summary:" -ForegroundColor Yellow
$gpuStats = ssh $server 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits'
$lines = $gpuStats -split "`n"
$totalUtil = 0
$totalMem = 0
foreach ($line in $lines) {
    $parts = $line.Split(',').Trim()
    if ($parts.Length -eq 2) {
        $totalUtil += [int]$parts[0]
        $totalMem += [int]$parts[1]
    }
}
$avgUtil = [math]::Round($totalUtil / 12, 1)
$avgMem = [math]::Round($totalMem / 12, 0)
Write-Host "  Average GPU Utilization: $avgUtil%" -ForegroundColor $(if ($avgUtil -gt 50) { "Green" } else { "Yellow" })
Write-Host "  Average GPU Memory: $avgMem MB" -ForegroundColor Cyan
Write-Host ""
Write-Host "Run this script again anytime to check status." -ForegroundColor Gray
Write-Host ""
