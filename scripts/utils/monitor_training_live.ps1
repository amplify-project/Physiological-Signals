#!/usr/bin/env pwsh
# Live training dashboard with auto-refresh every 5 seconds

function Get-TrainingMetrics {
    param($logContent)
    
    $metrics = @{
        Epoch = "N/A"
        TrainLoss = "N/A"
        TrainAcc = "N/A"
        ValLoss = "N/A"
        ValAcc = "N/A"
        LR = "N/A"
        BestValAcc = "N/A"
        Progress = "N/A"
    }
    
    # Extract latest epoch number
    if ($logContent -match "EPOCH (\d+)/(\d+)") {
        $metrics.Epoch = "$($matches[1])/$($matches[2])"
    }
    
    # Extract training metrics
    if ($logContent -match "Train Loss: ([\d\.]+) \| Train Acc: ([\d\.]+)") {
        $metrics.TrainLoss = $matches[1]
        $metrics.TrainAcc = $matches[2]
    }
    
    # Extract validation metrics
    if ($logContent -match "Val Loss: ([\d\.]+) \| Val Acc: ([\d\.]+)") {
        $metrics.ValLoss = $matches[1]
        $metrics.ValAcc = $matches[2]
    }
    
    # Extract learning rate
    if ($logContent -match "LR: ([\d\.e\-]+)") {
        $metrics.LR = $matches[1]
    }
    
    # Extract best validation accuracy
    if ($logContent -match "Saved best model \(val_acc: ([\d\.]+)\)") {
        $metrics.BestValAcc = $matches[1]
    }
    
    # Extract progress bar info
    if ($logContent -match "Training:\s+(\d+)%") {
        $metrics.Progress = "$($matches[1])%"
    }
    
    return $metrics
}

function Draw-ProgressBar {
    param([int]$percent, [int]$width = 10)
    $filled = [math]::Floor($percent / 10)
    $empty = $width - $filled
    return "[" + ("█" * $filled) + (" " * $empty) + "]"
}

function Show-Dashboard {
    param($serverIP = "192.168.200.206")
    
    # Get process info
    $processInfo = ssh $serverIP "ps aux | grep 'train_action_transformer_dataparallel.py' | grep -v grep | head -1" 2>$null
    $isRunning = $processInfo -ne $null -and $processInfo -ne ""
    
    # Get log content (last 100 lines)
    $logContent = ssh $serverIP "tail -100 ~/concert_engagement/logs/training_kinetics_transformer_dataparallel.log 2>/dev/null" 2>$null
    
    # Parse metrics
    $metrics = Get-TrainingMetrics -logContent $logContent
    
    # Get GPU stats
    $gpuStats = ssh $serverIP "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits" 2>$null
    
    # Clear screen and draw
    Clear-Host
    
    $timestamp = Get-Date -Format "HH:mm:ss"
    
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Kinetics-700 Action Transformer Training Monitor" -ForegroundColor White
    Write-Host "  Server: $serverIP | $timestamp" -ForegroundColor Gray
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    if ($isRunning) {
        Write-Host "✅ Training Process: RUNNING" -ForegroundColor Green
    } else {
        Write-Host "❌ Training Process: NOT RUNNING" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "───────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "  TRAINING METRICS" -ForegroundColor Yellow
    Write-Host "───────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host ("  Epoch:           " + $metrics.Epoch.PadRight(30) + "Progress:     " + $metrics.Progress) -ForegroundColor White
    Write-Host ("  Train Loss:      " + $metrics.TrainLoss.PadRight(30) + "Train Acc:    " + $metrics.TrainAcc) -ForegroundColor White
    Write-Host ("  Val Loss:        " + $metrics.ValLoss.PadRight(30) + "Val Acc:      " + $metrics.ValAcc) -ForegroundColor White
    Write-Host ("  Learning Rate:   " + $metrics.LR.PadRight(30) + "Best Val Acc: " + $metrics.BestValAcc) -ForegroundColor Cyan
    
    Write-Host ""
    Write-Host "───────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "  GPU UTILIZATION (12x Tesla T4)" -ForegroundColor Yellow
    Write-Host "───────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    
    if ($gpuStats) {
        foreach ($line in $gpuStats -split "`n") {
            if ($line.Trim()) {
                $parts = $line -split ","
                $idx = $parts[0].Trim()
                $util = [int]$parts[1].Trim()
                $memUsed = [int]$parts[2].Trim()
                $memTotal = [int]$parts[3].Trim()
                $temp = $parts[4].Trim()
                
                $memPercent = [math]::Round(($memUsed / $memTotal) * 100)
                $bar = Draw-ProgressBar -percent $util
                
                $color = if ($util -gt 80) { "Green" } elseif ($util -gt 20) { "Yellow" } else { "Gray" }
                $line = "  GPU $($idx.PadLeft(2)): $bar $($util.ToString().PadLeft(3))%  |  Mem: $($memUsed.ToString().PadLeft(5))/$memTotal MB ($($memPercent)%)  |  $temp°C"
                Write-Host $line -ForegroundColor $color
            }
        }
    } else {
        Write-Host "  Could not retrieve GPU stats" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "───────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "  Refreshing in 5 seconds... (Press Ctrl+C to exit)" -ForegroundColor Gray
    Write-Host "───────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
}

# Main loop
Write-Host "Starting live training monitor..." -ForegroundColor Cyan
Start-Sleep -Seconds 1

while ($true) {
    try {
        Show-Dashboard
        Start-Sleep -Seconds 5
    } catch {
        Write-Host "`nMonitoring stopped." -ForegroundColor Yellow
        break
    }
}
