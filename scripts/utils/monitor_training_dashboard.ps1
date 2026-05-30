#!/usr/bin/env pwsh
# Real-time training monitoring dashboard

$server = "eoghan@192.168.200.206"

function Get-TrainingMetrics {
    $log = ssh $server 'tail -50 ~/concert_engagement/logs/training_kinetics_transformer_ddp.log 2>/dev/null'
    
    # Extract last epoch info
    $epochLine = $log | Select-String "Epoch (\d+)/(\d+)" | Select-Object -Last 1
    $trainLine = $log | Select-String "Train Loss: ([\d\.]+) \| Train Acc: ([\d\.]+)" | Select-Object -Last 1
    $valLine = $log | Select-String "Val Loss: ([\d\.]+) \| Val Acc: ([\d\.]+)" | Select-Object -Last 1
    $lrLine = $log | Select-String "LR: ([\d\.e\-]+)" | Select-Object -Last 1
    $savedLine = $log | Select-String "Saved best model \(val_acc: ([\d\.]+)\)" | Select-Object -Last 1
    
    return @{
        Epoch = if ($epochLine) { $epochLine.Matches.Groups[1].Value + "/" + $epochLine.Matches.Groups[2].Value } else { "N/A" }
        TrainLoss = if ($trainLine) { $trainLine.Matches.Groups[1].Value } else { "N/A" }
        TrainAcc = if ($trainLine) { ([float]$trainLine.Matches.Groups[2].Value * 100).ToString("F2") + "%" } else { "N/A" }
        ValLoss = if ($valLine) { $valLine.Matches.Groups[1].Value } else { "N/A" }
        ValAcc = if ($valLine) { ([float]$valLine.Matches.Groups[2].Value * 100).ToString("F2") + "%" } else { "N/A" }
        LR = if ($lrLine) { $lrLine.Matches.Groups[1].Value } else { "N/A" }
        BestAcc = if ($savedLine) { ([float]$savedLine.Matches.Groups[1].Value * 100).ToString("F2") + "%" } else { "N/A" }
    }
}

function Get-GPUStats {
    $gpu = ssh $server 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits'
    return $gpu
}

function Get-ProcessInfo {
    $proc = ssh $server 'ps aux | grep train_action_transformer_ddp | grep -v grep | head -1'
    return $proc
}

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Kinetics-700 Action Transformer Training Monitor" -ForegroundColor Cyan
Write-Host "  Server: 192.168.200.206 | Press Ctrl+C to exit" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

while ($true) {
    Clear-Host
    
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Kinetics-700 Action Transformer Training Monitor" -ForegroundColor Cyan
    Write-Host "  Server: 192.168.200.206 | $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    # Process info
    $proc = Get-ProcessInfo
    if ($proc) {
        Write-Host "✅ Training Process: RUNNING" -ForegroundColor Green
        Write-Host "   PID: $($proc.Split()[1])" -ForegroundColor Gray
    } else {
        Write-Host "❌ Training Process: NOT RUNNING" -ForegroundColor Red
    }
    Write-Host ""
    
    # Training metrics
    $metrics = Get-TrainingMetrics
    Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "  TRAINING METRICS" -ForegroundColor Yellow
    Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "  Epoch:           $($metrics.Epoch)" -ForegroundColor White
    Write-Host "  Train Loss:      $($metrics.TrainLoss)" -ForegroundColor White
    Write-Host "  Train Accuracy:  $($metrics.TrainAcc)" -ForegroundColor Green
    Write-Host "  Val Loss:        $($metrics.ValLoss)" -ForegroundColor White
    Write-Host "  Val Accuracy:    $($metrics.ValAcc)" -ForegroundColor Green
    Write-Host "  Learning Rate:   $($metrics.LR)" -ForegroundColor Cyan
    Write-Host "  Best Val Acc:    $($metrics.BestAcc)" -ForegroundColor Magenta
    Write-Host ""
    
    # GPU stats
    Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "  GPU UTILIZATION (12x Tesla T4)" -ForegroundColor Yellow
    Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    
    $gpus = Get-GPUStats
    foreach ($gpu in $gpus) {
        $parts = $gpu.Split(',').Trim()
        $idx = $parts[0]
        $util = $parts[1]
        $memUsed = $parts[2]
        $memTotal = $parts[3]
        $temp = $parts[4]
        
        $utilBar = "█" * [math]::Floor($util / 10)
        $utilBar = $utilBar.PadRight(10)
        
        $memPct = [math]::Round(([int]$memUsed / [int]$memTotal) * 100)
        
        $color = if ($util -gt 80) { "Green" } elseif ($util -gt 50) { "Yellow" } else { "DarkGray" }
        
        Write-Host "  GPU $($idx.PadLeft(2)): [$utilBar] $($util.PadLeft(3))%  |  Mem: $($memUsed.PadLeft(5))/$($memTotal) MB ($($memPct)%)  |  $temp°C" -ForegroundColor $color
    }
    
    Write-Host ""
    Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "  Refreshing in 5 seconds... (Press Ctrl+C to exit)" -ForegroundColor DarkGray
    Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    
    Start-Sleep -Seconds 5
}
