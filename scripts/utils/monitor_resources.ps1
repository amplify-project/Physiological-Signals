# Real-time GPU and CPU Monitoring Dashboard
# Simple, clean display of all resource usage

param(
    [string]$Server = "192.168.200.206",
    [string]$User = "eoghan",
    [int]$IntervalSeconds = 5
)

function Get-ResourceStats {
    param([hashtable]$PrevStats = $null)
    
    # Get GPU stats
    $gpuData = ssh ${User}@${Server} "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
    
    # Get overall CPU usage
    $cpuTotal = ssh ${User}@${Server} "top -bn1 | grep 'Cpu(s)' | awk '{print 100 - `$8}'"
    
    # Get per-core CPU usage (first 12 cores to match GPU count)
    $cpuCores = ssh ${User}@${Server} "mpstat -P ALL 1 1 | awk '/Average:/ && `$2 ~ /^[0-9]+`$/ {printf `"%s:%.1f\n`", `$2, 100-`$NF}' | head -12"
    
    # Get memory
    $memory = ssh ${User}@${Server} "free -g | awk 'NR==2{printf `"%.0f/%.0f GB (%.0f%%)`", `$3, `$2, `$3*100/`$2}'"
    
    # Get extraction process count
    $processes = ssh ${User}@${Server} "ps aux | grep 'python.*extract_kinetics' | grep -v grep | wc -l"
    
    # Get extracted file count and calculate rate
    $extracted = ssh ${User}@${Server} "find ~/concert_engagement/data/processed/features_kinetics700 -name '*.npz' -type f 2>/dev/null | wc -l"
    $currentTime = Get-Date
    
    $rate = 0.0
    if ($PrevStats -and $PrevStats.Extracted -and $PrevStats.Time) {
        $timeDiff = ($currentTime - $PrevStats.Time).TotalSeconds
        $fileDiff = [int]$extracted - $PrevStats.Extracted
        if ($timeDiff -gt 0 -and $fileDiff -gt 0) {
            $rate = $fileDiff / $timeDiff
        }
    }
    
    return @{
        GPUData = $gpuData
        CPUTotal = [float]$cpuTotal
        CPUCores = $cpuCores
        Memory = $memory
        Processes = [int]$processes
        Extracted = [int]$extracted
        Time = $currentTime
        Rate = $rate
    }
}

function Draw-Bar {
    param([float]$Value, [int]$Width = 20)
    
    $filled = [math]::Floor(($Value / 100) * $Width)
    $empty = $Width - $filled
    
    $color = if ($Value -gt 80) { "Red" } 
             elseif ($Value -gt 50) { "Yellow" } 
             elseif ($Value -gt 10) { "Green" }
             else { "Gray" }
    
    Write-Host "[" -NoNewline
    Write-Host ("█" * $filled) -ForegroundColor $color -NoNewline
    Write-Host ("░" * $empty) -ForegroundColor DarkGray -NoNewline
    Write-Host "]" -NoNewline
}

Clear-Host
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║              🎸 GPU & CPU MONITORING DASHBOARD                                 ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$iteration = 0
$prevStats = $null
while ($true) {
    $iteration++
    
    # Move cursor back to top (keep header)
    if ($iteration -gt 1) {
        $pos = $host.UI.RawUI.CursorPosition
        $pos.Y = 4
        $host.UI.RawUI.CursorPosition = $pos
    }
    
    try {
        $stats = Get-ResourceStats -PrevStats $prevStats
        
        # Header info
        $rateDisplay = if ($stats.Rate -gt 0) { "{0:N2} files/sec" -f $stats.Rate } else { "Calculating..." }
        Write-Host ("Update #{0,-5} at {1}" -f $iteration, (Get-Date -Format "HH:mm:ss")) -ForegroundColor Gray
        Write-Host ("Extracted: {0,6:N0} files  |  Rate: {1,-20}  |  Processes: {2}  |  Memory: {3}" -f $stats.Extracted, $rateDisplay, $stats.Processes, $stats.Memory) -ForegroundColor Cyan
        Write-Host ""
        
        # GPU Section
        Write-Host "🚀 GPU UTILIZATION (12x Tesla T4)" -ForegroundColor Yellow
        Write-Host ("─" * 80) -ForegroundColor DarkGray
        
        if ($stats.GPUData) {
            $gpuLines = $stats.GPUData -split "`n" | Where-Object { $_.Trim() }
            foreach ($line in $gpuLines) {
                $parts = $line -split ','
                if ($parts.Count -ge 5) {
                    $gpuId = $parts[0].Trim()
                    $gpuUtil = [float]$parts[1].Trim()
                    $memUsed = [int]$parts[2].Trim()
                    $memTotal = [int]$parts[3].Trim()
                    $temp = [int]$parts[4].Trim()
                    
                    $memPct = ($memUsed / $memTotal * 100)
                    
                    Write-Host ("GPU {0,2}: " -f $gpuId) -NoNewline
                    Draw-Bar -Value $gpuUtil -Width 30
                    Write-Host (" {0,5:N1}%  |  Mem: {1,5:N0}MB ({2,3:N0}%)  |  {3,2}°C" -f $gpuUtil, $memUsed, $memPct, $temp)
                }
            }
        }
        
        Write-Host ""
        
        # CPU Section
        Write-Host "💻 CPU UTILIZATION (104 cores total)" -ForegroundColor Yellow
        Write-Host ("─" * 80) -ForegroundColor DarkGray
        Write-Host ("Overall: ") -NoNewline
        Draw-Bar -Value $stats.CPUTotal -Width 50
        Write-Host (" {0,5:N1}%" -f $stats.CPUTotal)
        
        Write-Host ""
        Write-Host "First 12 Cores (matching GPU workers):" -ForegroundColor Gray
        if ($stats.CPUCores) {
            $coreLines = $stats.CPUCores -split "`n" | Where-Object { $_.Trim() }
            $row = 0
            foreach ($line in $coreLines) {
                $parts = $line -split ':'
                if ($parts.Count -ge 2) {
                    $coreId = $parts[0].Trim()
                    $coreUtil = [float]$parts[1].Trim()
                    
                    Write-Host ("  Core {0,2}: " -f $coreId) -NoNewline
                    Draw-Bar -Value $coreUtil -Width 20
                    Write-Host (" {0,5:N1}%  " -f $coreUtil) -NoNewline
                    
                    # 3 cores per row
                    if (($row + 1) % 3 -eq 0) {
                        Write-Host ""
                    }
                    $row++
                }
            }
            if ($row % 3 -ne 0) { Write-Host "" }
        }
        
        Write-Host ""
        Write-Host ("─" * 80) -ForegroundColor DarkGray
        Write-Host ("Refreshing every {0} seconds (Ctrl+C to stop)" -f $IntervalSeconds) -ForegroundColor Gray
        Write-Host ""
        
        # Store current stats for next iteration
        $prevStats = $stats
        
        Start-Sleep -Seconds $IntervalSeconds
    }
    catch {
        Write-Host "Error: $_" -ForegroundColor Red
        Start-Sleep -Seconds $IntervalSeconds
    }
}
