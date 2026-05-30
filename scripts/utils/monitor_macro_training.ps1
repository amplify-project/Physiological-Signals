#==================================================================================
# Training Monitor Dashboard - Macro Classifier (13 Classes)
#==================================================================================
# Purpose: Real-time monitoring of 12-GPU distributed training
# Updates:  Every 30 seconds
# Usage:    .\scripts\monitor_macro_training.ps1
#==================================================================================

param(
    [string]$Server = "eoghan@192.168.200.206",
    [string]$LogFile = "logs/training_12gpus_macro.log",
    [int]$RefreshInterval = 30
)

function Get-TrainingStats {
    param([string]$Server, [string]$LogFile)
    
    # Get last 100 lines of log
    $logContent = ssh $Server "tail -100 ~/concert_engagement/$LogFile 2>/dev/null"
    
    # Parse current epoch and batch
    $epochMatch = $logContent | Select-String "EPOCH (\d+)/(\d+)" | Select-Object -Last 1
    $currentEpoch = if ($epochMatch) { $epochMatch.Matches.Groups[1].Value } else { "?" }
    $totalEpochs = if ($epochMatch) { $epochMatch.Matches.Groups[2].Value } else { "?" }
    
    # Parse training progress
    $progressMatch = $logContent | Select-String "Training:\s+(\d+)%.*?(\d+)/(\d+).*?loss=([\d.]+)" | Select-Object -Last 1
    $trainProgress = if ($progressMatch) { $progressMatch.Matches.Groups[1].Value } else { "?" }
    $currentBatch = if ($progressMatch) { $progressMatch.Matches.Groups[2].Value } else { "?" }
    $totalBatches = if ($progressMatch) { $progressMatch.Matches.Groups[3].Value } else { "?" }
    $currentLoss = if ($progressMatch) { $progressMatch.Matches.Groups[4].Value } else { "?" }
    
    # Parse validation accuracy (if available)
    $valMatch = $logContent | Select-String "Val Acc:\s+([\d.]+)%" | Select-Object -Last 1
    $valAcc = if ($valMatch) { $valMatch.Matches.Groups[1].Value } else { "N/A" }
    
    # Parse best accuracy
    $bestMatch = $logContent | Select-String "🏆.*?([\d.]+)%" | Select-Object -Last 1
    $bestAcc = if ($bestMatch) { $bestMatch.Matches.Groups[1].Value } else { "N/A" }
    
    # Get last log line
    $lastLine = $logContent | Select-Object -Last 1
    
    return @{
        CurrentEpoch = $currentEpoch
        TotalEpochs = $totalEpochs
        TrainProgress = $trainProgress
        CurrentBatch = $currentBatch
        TotalBatches = $totalBatches
        CurrentLoss = $currentLoss
        ValAcc = $valAcc
        BestAcc = $bestAcc
        LastLine = $lastLine
    }
}

function Get-GPUStats {
    param([string]$Server)
    
    $gpuData = ssh $Server "nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits"
    
    $gpus = @()
    foreach ($line in $gpuData) {
        $parts = $line -split ','
        $gpus += @{
            Index = $parts[0].Trim()
            Temp = $parts[1].Trim()
            Util = $parts[2].Trim()
            MemUsed = $parts[3].Trim()
            MemTotal = $parts[4].Trim()
            Power = $parts[5].Trim()
        }
    }
    
    return $gpus
}

function Get-SystemStats {
    param([string]$Server)
    
    # Get CPU and memory stats
    $sysData = ssh $Server @"
top -bn1 | grep 'Cpu(s)' | awk '{print \$2}' | cut -d'%' -f1
free -m | awk 'NR==2{printf "%.1f,%.1f", \$3/1024, \$2/1024}'
ps aux | grep -E 'torchrun|train_action' | grep -v grep | wc -l
"@
    
    $lines = $sysData -split "`n"
    $cpuUsage = if ($lines[0]) { [math]::Round([float]$lines[0], 1) } else { 0 }
    $memParts = if ($lines[1]) { $lines[1] -split ',' } else { @(0, 0) }
    $processCount = if ($lines[2]) { [int]$lines[2] } else { 0 }
    
    return @{
        CPUUsage = $cpuUsage
        MemUsedGB = [math]::Round([float]$memParts[0], 1)
        MemTotalGB = [math]::Round([float]$memParts[1], 1)
        ProcessCount = $processCount
    }
}

function Show-Dashboard {
    param($TrainStats, $GPUStats, $SysStats, $UpdateTime)
    
    Clear-Host
    
    # Header
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "                    MACRO TRAINING MONITOR (13 Classes)                        " -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    # Training Progress
    Write-Host "┌─────────────────────────────────────────────────────────────────────────────┐" -ForegroundColor DarkGray
    Write-Host "│ " -NoNewline -ForegroundColor DarkGray
    Write-Host "TRAINING PROGRESS                                                           " -NoNewline -ForegroundColor Yellow
    Write-Host "│" -ForegroundColor DarkGray
    Write-Host "└─────────────────────────────────────────────────────────────────────────────┘" -ForegroundColor DarkGray
    
    Write-Host "  Epoch:          " -NoNewline -ForegroundColor Gray
    Write-Host "$($TrainStats.CurrentEpoch)/$($TrainStats.TotalEpochs)" -ForegroundColor White
    
    Write-Host "  Batch:          " -NoNewline -ForegroundColor Gray
    Write-Host "$($TrainStats.CurrentBatch)/$($TrainStats.TotalBatches) ($($TrainStats.TrainProgress)%)" -ForegroundColor White
    
    Write-Host "  Current Loss:   " -NoNewline -ForegroundColor Gray
    $lossColor = if ($TrainStats.CurrentLoss -ne "?") {
        $loss = [float]$TrainStats.CurrentLoss
        if ($loss -lt 1.5) { "Green" } elseif ($loss -lt 2.0) { "Yellow" } else { "Red" }
    } else { "White" }
    Write-Host $TrainStats.CurrentLoss -ForegroundColor $lossColor
    
    Write-Host "  Val Accuracy:   " -NoNewline -ForegroundColor Gray
    $valColor = if ($TrainStats.ValAcc -ne "N/A") {
        $acc = [float]$TrainStats.ValAcc
        if ($acc -gt 60) { "Green" } elseif ($acc -gt 50) { "Yellow" } else { "Red" }
    } else { "White" }
    Write-Host "$($TrainStats.ValAcc)$(if ($TrainStats.ValAcc -ne 'N/A') {'%'})" -ForegroundColor $valColor
    
    Write-Host "  Best Accuracy:  " -NoNewline -ForegroundColor Gray
    Write-Host "$($TrainStats.BestAcc)$(if ($TrainStats.BestAcc -ne 'N/A') {'%'})" -ForegroundColor Green
    
    Write-Host ""
    
    # GPU Stats
    Write-Host "┌─────────────────────────────────────────────────────────────────────────────┐" -ForegroundColor DarkGray
    Write-Host "│ " -NoNewline -ForegroundColor DarkGray
    Write-Host "GPU STATUS (12x Tesla T4)                                                  " -NoNewline -ForegroundColor Yellow
    Write-Host "│" -ForegroundColor DarkGray
    Write-Host "└─────────────────────────────────────────────────────────────────────────────┘" -ForegroundColor DarkGray
    
    Write-Host "  GPU  Temp   Util    Memory        Power" -ForegroundColor Gray
    Write-Host "  ───  ────   ────    ──────        ─────" -ForegroundColor DarkGray
    
    foreach ($gpu in $GPUStats) {
        $tempColor = if ([int]$gpu.Temp -gt 70) { "Red" } elseif ([int]$gpu.Temp -gt 60) { "Yellow" } else { "Green" }
        $utilColor = if ([int]$gpu.Util -gt 80) { "Green" } elseif ([int]$gpu.Util -gt 50) { "Yellow" } else { "White" }
        $memPercent = [math]::Round(([float]$gpu.MemUsed / [float]$gpu.MemTotal) * 100, 0)
        $memBar = "█" * [math]::Floor($memPercent / 10) + "░" * (10 - [math]::Floor($memPercent / 10))
        
        Write-Host "  " -NoNewline
        Write-Host ("{0,2}" -f $gpu.Index) -NoNewline -ForegroundColor White
        Write-Host "   " -NoNewline
        Write-Host ("{0,3}°C" -f $gpu.Temp) -NoNewline -ForegroundColor $tempColor
        Write-Host "  " -NoNewline
        Write-Host ("{0,3}%" -f $gpu.Util) -NoNewline -ForegroundColor $utilColor
        Write-Host "   " -NoNewline
        Write-Host $memBar -NoNewline -ForegroundColor Cyan
        Write-Host (" {0,2}%" -f $memPercent) -NoNewline -ForegroundColor Gray
        Write-Host "    " -NoNewline
        Write-Host ("{0,3}W" -f $gpu.Power) -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    # System Stats
    Write-Host "┌─────────────────────────────────────────────────────────────────────────────┐" -ForegroundColor DarkGray
    Write-Host "│ " -NoNewline -ForegroundColor DarkGray
    Write-Host "SYSTEM RESOURCES                                                            " -NoNewline -ForegroundColor Yellow
    Write-Host "│" -ForegroundColor DarkGray
    Write-Host "└─────────────────────────────────────────────────────────────────────────────┘" -ForegroundColor DarkGray
    
    Write-Host "  CPU Usage:      " -NoNewline -ForegroundColor Gray
    Write-Host "$($SysStats.CPUUsage)%" -ForegroundColor $(if ($SysStats.CPUUsage -gt 80) { "Red" } else { "Green" })
    
    Write-Host "  Memory:         " -NoNewline -ForegroundColor Gray
    Write-Host "$($SysStats.MemUsedGB) GB / $($SysStats.MemTotalGB) GB" -ForegroundColor White
    
    Write-Host "  Training Procs: " -NoNewline -ForegroundColor Gray
    Write-Host $SysStats.ProcessCount -ForegroundColor $(if ($SysStats.ProcessCount -eq 13) { "Green" } else { "Yellow" })
    
    Write-Host ""
    
    # Last Log Line
    Write-Host "┌─────────────────────────────────────────────────────────────────────────────┐" -ForegroundColor DarkGray
    Write-Host "│ " -NoNewline -ForegroundColor DarkGray
    Write-Host "LATEST LOG                                                                  " -NoNewline -ForegroundColor Yellow
    Write-Host "│" -ForegroundColor DarkGray
    Write-Host "└─────────────────────────────────────────────────────────────────────────────┘" -ForegroundColor DarkGray
    
    $lastLineDisplay = if ($TrainStats.LastLine.Length -gt 75) { 
        $TrainStats.LastLine.Substring(0, 72) + "..."
    } else { 
        $TrainStats.LastLine.PadRight(75)
    }
    Write-Host "  $lastLineDisplay" -ForegroundColor Cyan
    
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Last Update: $UpdateTime" -NoNewline -ForegroundColor Gray
    Write-Host " | Next refresh in $RefreshInterval seconds | Press Ctrl+C to exit" -ForegroundColor DarkGray
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
}

# Main loop
Write-Host "Starting training monitor..." -ForegroundColor Green
Write-Host "Connecting to $Server..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

while ($true) {
    try {
        $updateTime = Get-Date -Format "HH:mm:ss"
        
        # Gather all stats
        $trainStats = Get-TrainingStats -Server $Server -LogFile $LogFile
        $gpuStats = Get-GPUStats -Server $Server
        $sysStats = Get-SystemStats -Server $Server
        
        # Display dashboard
        Show-Dashboard -TrainStats $trainStats -GPUStats $gpuStats -SysStats $sysStats -UpdateTime $updateTime
        
        # Wait for next update
        Start-Sleep -Seconds $RefreshInterval
    }
    catch {
        Write-Host "Error: $_" -ForegroundColor Red
        Write-Host "Retrying in $RefreshInterval seconds..." -ForegroundColor Yellow
        Start-Sleep -Seconds $RefreshInterval
    }
}
