# Launch Binary Training with Live Monitoring
# This script handles VPN reconnection and shows immediate feedback

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Binary Training Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to run SSH command with retry
function Invoke-SSHWithRetry {
    param([string]$Command, [int]$MaxRetries = 3)
    
    for ($i = 1; $i -le $MaxRetries; $i++) {
        Write-Host "Attempt $i of $MaxRetries..." -ForegroundColor Yellow
        try {
            $result = ssh eoghan@192.168.200.206 $Command 2>&1
            if ($LASTEXITCODE -eq 0) {
                return $result
            }
        } catch {
            Write-Host "Connection failed: $_" -ForegroundColor Red
        }
        
        if ($i -lt $MaxRetries) {
            Write-Host "Retrying in 2 seconds..." -ForegroundColor Yellow
            Start-Sleep -Seconds 2
        }
    }
    
    Write-Host "Failed after $MaxRetries attempts" -ForegroundColor Red
    return $null
}

# Step 1: Create logs directory
Write-Host "[1/4] Creating logs directory..." -ForegroundColor Green
Invoke-SSHWithRetry "mkdir -p ~/concert_engagement/logs"

# Step 2: Start training with nohup
Write-Host "[2/4] Starting binary training with nohup..." -ForegroundColor Green
$startCmd = "cd ~/concert_engagement && source venv/bin/activate && nohup python scripts/train_binary.py --features-dir data/processed/features_kinetics700 --epochs 50 --batch-size 32 --lr 0.0001 > logs/binary_training.log 2>&1 & echo 'Training process started'"
Invoke-SSHWithRetry $startCmd

Write-Host ""
Write-Host "Training launched! Waiting 3 seconds for initialization..." -ForegroundColor Green
Start-Sleep -Seconds 3

# Step 3: Verify process is running
Write-Host "[3/4] Verifying training process..." -ForegroundColor Green
$process = Invoke-SSHWithRetry "ps aux | grep 'train_binary.py' | grep -v grep"
if ($process) {
    Write-Host "✓ Training process is running!" -ForegroundColor Green
    Write-Host $process
} else {
    Write-Host "✗ Training process not found. Checking logs..." -ForegroundColor Red
}

Write-Host ""
Write-Host "[4/4] Starting live log monitor..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 4: Monitor logs with auto-reconnect
while ($true) {
    try {
        Write-Host "Connecting to log stream..." -ForegroundColor Yellow
        ssh eoghan@192.168.200.206 "tail -f ~/concert_engagement/logs/binary_training.log"
    } catch {
        Write-Host ""
        Write-Host "Connection lost. Reconnecting in 5 seconds..." -ForegroundColor Red
        Start-Sleep -Seconds 5
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Connection lost. Reconnecting in 5 seconds..." -ForegroundColor Red
        Start-Sleep -Seconds 5
    }
}
