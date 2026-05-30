# Live Extraction Monitoring Dashboard
$server = "eoghan@192.168.200.206"

Write-Host "=== KINETICS-700 FEATURE EXTRACTION MONITOR ===" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop monitoring`n" -ForegroundColor Yellow

while ($true) {
    Clear-Host
    $timestamp = Get-Date -Format "HH:mm:ss"
    
    Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║  KINETICS-700 MEDIAPIPE FEATURE EXTRACTION - $timestamp  ║" -ForegroundColor Cyan
    Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    
    # Get extraction progress
    $featureCount = ssh $server "find ~/concert_engagement/data/processed/Kinetics700_features -name '*.npz' 2>/dev/null | wc -l"
    $totalVideos = ssh $server "find ~/concert_engagement/data/raw/Kinetics700 -name '*.mp4' 2>/dev/null | wc -l"
    
    $progress = [math]::Round(($featureCount / $totalVideos) * 100, 2)
    $remaining = $totalVideos - $featureCount
    
    Write-Host "📊 PROGRESS" -ForegroundColor Green
    Write-Host "  Features Extracted: $featureCount / $totalVideos" -ForegroundColor White
    Write-Host "  Progress: $progress%" -ForegroundColor White
    Write-Host "  Remaining: $remaining videos" -ForegroundColor White
    Write-Host ""
    
    # GPU utilization
    Write-Host "🎮 GPU UTILIZATION (12x Tesla T4)" -ForegroundColor Green
    $gpuStats = ssh $server "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader,nounits"
    
    $totalUtil = 0
    $gpuLines = $gpuStats -split "`n"
    foreach ($line in $gpuLines) {
        if ($line -match '(\d+),\s*([^,]+),\s*(\d+),\s*(\d+)') {
            $gpuId = $matches[1]
            $util = $matches[3]
            $mem = $matches[4]
            $totalUtil += [int]$util
            
            $bar = "█" * ([math]::Floor($util / 5))
            $color = if ($util -gt 70) { "Green" } elseif ($util -gt 30) { "Yellow" } else { "Gray" }
            Write-Host "  GPU $gpuId : " -NoNewline
            Write-Host "$bar" -ForegroundColor $color -NoNewline
            Write-Host " $util% (${mem}MB)" -ForegroundColor White
        }
    }
    
    $avgUtil = [math]::Round($totalUtil / 12, 1)
    Write-Host "  Average: $avgUtil%" -ForegroundColor Cyan
    Write-Host ""
    
    # CPU Cores for GPU 0 and 1 processes
    Write-Host "💻 CPU CORES (GPU 0 & 1 Workers)" -ForegroundColor Green
    $cpuInfo = ssh $server "ps aux | grep 'python.*extract_kinetics' | grep -v grep | head -2 | awk '{print `$3}'"
    if ($cpuInfo) {
        $cpuLines = $cpuInfo -split "`n"
        $gpu0cpu = if ($cpuLines[0]) { $cpuLines[0] } else { "0.0" }
        $gpu1cpu = if ($cpuLines[1]) { $cpuLines[1] } else { "0.0" }
        Write-Host "  Worker 0 (GPU 0): $gpu0cpu% CPU" -ForegroundColor White
        Write-Host "  Worker 1 (GPU 1): $gpu1cpu% CPU" -ForegroundColor White
    }
    Write-Host ""
    
    # Processing rate
    if ($featureCount -gt 0) {
        $processInfo = ssh $server "ps aux | grep 'extract_kinetics_features' | grep -v grep | awk '{print `$10}'"
        if ($processInfo) {
            $runtime = $processInfo
            Write-Host "⚡ PROCESSING INFO" -ForegroundColor Green
            Write-Host "  Runtime: $runtime minutes" -ForegroundColor White
            
            if ($avgUtil -gt 10) {
                $eta = [math]::Round(($remaining / $featureCount) * $runtime, 0)
                Write-Host "  Estimated Time Remaining: ~$eta minutes" -ForegroundColor Yellow
            }
        }
    }
    
    Write-Host ""
    Write-Host "Next update in 10 seconds..." -ForegroundColor Gray
    Start-Sleep -Seconds 10
}
