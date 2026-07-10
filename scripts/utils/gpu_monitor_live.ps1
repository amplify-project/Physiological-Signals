#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Live console GPU monitor for the sri-gpu-12t4 server (192.168.200.206).

.DESCRIPTION
    Polls all 12 GPUs every N minutes (default 5) and prints a bar-graph of
    utilisation, temperature, memory, and the USERNAME(s) running on each GPU.

    Runs entirely in your own terminal via SSH key auth — it does NOT use
    Copilot, so it costs no tokens/credits. Press Ctrl+C to stop.

.PARAMETER IntervalMinutes
    Minutes between polls (default 5).

.PARAMETER Server
    SSH target (default eoghan@192.168.200.206).

.EXAMPLE
    pwsh -File scripts/utils/gpu_monitor_live.ps1
    pwsh -File scripts/utils/gpu_monitor_live.ps1 -IntervalMinutes 2
#>
param(
    [int]$IntervalMinutes = 5,
    [string]$Server = "eoghan@192.168.200.206"
)

# Single-line remote command (concatenated) with sectional markers.
# `$ escapes keep the $ for the remote bash, not PowerShell.
$remoteCmd = `
    "echo GPUSTART; " + `
    "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits; " + `
    "echo MAPSTART; " + `
    "nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader; " + `
    "echo APPSTART; " + `
    "nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | while IFS=, read bus pid; do u=`$(ps -o user= -p `$(echo `$pid | tr -d ' ') 2>/dev/null | tr -d ' '); echo `$bus'|'`$u; done; " + `
    "echo TRAINSTART; " + `
    "cd ~/concert_engagement 2>/dev/null; " + `
    "L=`$(ls -t logs/train_young_families_*.log 2>/dev/null | head -1); " + `
    "if [ -n ""`$L"" ]; then " + `
    "echo E'|'`$(grep -oE 'EPOCH [0-9]+/[0-9]+' ""`$L"" | tail -1); " + `
    "echo P'|'`$(tr '\r' '\n' < ""`$L"" | grep -E 'Training:|Validation:' | tail -1 | sed 's/|/ /g' | tr -s ' '); " + `
    "echo B'|'`$(grep 'Saved best model' ""`$L"" | tail -1 | grep -oE 'acc: [0-9.]+'); " + `
    "echo A'|'`$(pgrep -f train_young_families.py | head -1); " + `
    "fi"

function Get-UtilColor([int]$util) {
    if ($util -gt 80) { return "Green" }
    elseif ($util -gt 30) { return "Yellow" }
    else { return "DarkGray" }
}

function Get-TempColor([int]$temp) {
    if ($temp -ge 80) { return "Red" }
    elseif ($temp -ge 70) { return "Yellow" }
    else { return "Green" }
}

Write-Host ""
Write-Host "GPU monitor -> $Server | every $IntervalMinutes min | Ctrl+C to stop" -ForegroundColor Cyan
Write-Host ""

while ($true) {
    $raw = ssh -o BatchMode=yes -o ConnectTimeout=10 $Server $remoteCmd 2>$null

    if (-not $raw) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] No response from $Server (unreachable or SSH failed)" -ForegroundColor Red
        Start-Sleep -Seconds ($IntervalMinutes * 60)
        continue
    }

    # Split output into sections by marker.
    $section = ""
    $gpuLines = @(); $mapLines = @(); $appLines = @(); $trainLines = @()
    foreach ($line in $raw) {
        switch -Regex ($line) {
            '^GPUSTART'   { $section = "gpu";   continue }
            '^MAPSTART'   { $section = "map";   continue }
            '^APPSTART'   { $section = "app";   continue }
            '^TRAINSTART' { $section = "train"; continue }
            default {
                switch ($section) {
                    "gpu"   { $gpuLines += $line }
                    "map"   { $mapLines += $line }
                    "app"   { $appLines += $line }
                    "train" { $trainLines += $line }
                }
            }
        }
    }

    # Parse training section (E|epoch  P|progress  B|acc  A|pid).
    $trEpoch = ""; $trProgress = ""; $trAcc = ""; $trPid = ""
    foreach ($t in $trainLines) {
        $k = $t.Split('|', 2)
        if ($k.Count -lt 2) { continue }
        switch ($k[0].Trim()) {
            "E" { $trEpoch = ($k[1].Trim() -replace 'EPOCH ', '') }
            "P" { $trProgress = (($k[1] -replace '[^\x20-\x7E]', '') -replace '\s+', ' ').Trim() }
            "B" { $trAcc = $k[1].Trim() }
            "A" { $trPid = $k[1].Trim() }
        }
    }

    # bus_id -> index
    $busToIdx = @{}
    foreach ($m in $mapLines) {
        $p = $m.Split(',')
        if ($p.Count -ge 2) { $busToIdx[$p[1].Trim()] = $p[0].Trim() }
    }
    # index -> list of users (via compute-apps bus|user)
    $users = @{}
    foreach ($a in $appLines) {
        $p = $a.Split('|')
        if ($p.Count -lt 2) { continue }
        $bus = $p[0].Trim(); $user = $p[1].Trim()
        if (-not $user) { continue }
        $idx = $busToIdx[$bus]
        if (-not $idx) { continue }
        if (-not $users.ContainsKey($idx)) { $users[$idx] = New-Object System.Collections.Generic.List[string] }
        if (-not $users[$idx].Contains($user)) { $users[$idx].Add($user) }
    }

    Clear-Host
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "  GPU STATUS  |  $Server  |  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "  GPU  Utilisation            Util   Mem (MiB)        Temp   User(s)" -ForegroundColor DarkGray
    Write-Host "  ---  --------------------   ----   --------------   ----   -------" -ForegroundColor DarkGray

    $freeCount = 0
    foreach ($line in $gpuLines) {
        $p = $line.Split(',')
        if ($p.Count -lt 5) { continue }
        $idx = $p[0].Trim(); $util = [int]$p[1].Trim(); $memUsed = [int]$p[2].Trim(); $memTotal = [int]$p[3].Trim(); $temp = [int]$p[4].Trim()

        $barLen = [math]::Floor($util / 5)              # 20-char bar (each = 5%)
        $bar = ("#" * $barLen).PadRight(20)
        $uColor = Get-UtilColor $util
        $tColor = Get-TempColor $temp

        $userList = if ($users.ContainsKey($idx)) { ($users[$idx] -join ",") } else { "-" }
        if ($util -lt 5 -and $memUsed -lt 200) { $freeCount++; if ($userList -eq "-") { $userList = "(free)" } }

        # Composite line: print the coloured bar, then the rest.
        Write-Host ("  {0,2}   " -f $idx) -NoNewline
        Write-Host ("[{0}]" -f $bar) -NoNewline -ForegroundColor $uColor
        Write-Host ("  {0,3}%   {1,6}/{2,-6}  " -f $util, $memUsed, $memTotal) -NoNewline
        Write-Host ("{0,3}C" -f $temp) -NoNewline -ForegroundColor $tColor
        Write-Host ("   {0}" -f $userList)
    }

    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host ("  Free GPUs (idle + empty): {0}" -f $freeCount) -ForegroundColor Green

    # Training progress panel.
    Write-Host ""
    Write-Host "  YOUNG-FAMILIES TRAINING" -ForegroundColor Cyan
    if ($trEpoch -or $trProgress) {
        $runState = if ($trPid) { "running (PID $trPid)" } else { "NOT RUNNING" }
        $runColor = if ($trPid) { "Green" } else { "Red" }
        Write-Host ("  Status : ") -NoNewline; Write-Host $runState -ForegroundColor $runColor
        if ($trEpoch) { Write-Host ("  Epoch  : {0}" -f $trEpoch) -ForegroundColor White }
        if ($trProgress) { Write-Host ("  Batch  : {0}" -f $trProgress) -ForegroundColor Gray }
        if ($trAcc) { Write-Host ("  Best   : {0}" -f $trAcc) -ForegroundColor Green }
    } else {
        Write-Host "  No active training log found." -ForegroundColor DarkGray
    }

    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host ("  Next refresh in {0} min  |  Ctrl+C to stop" -f $IntervalMinutes) -ForegroundColor DarkGray

    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
