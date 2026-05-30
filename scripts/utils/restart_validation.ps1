# Restart validation with updated error reporting script
# Kill old validation process and start new one with diagnostics

$ErrorActionPreference = "Continue"

Write-Host "`n🔄 Restarting validation with error diagnostics..." -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

# SSH command to kill old python validation and start new one
$sshCommand = @"
pkill -f validate_features.py
sleep 2
cd ~/concert_engagement
nohup python scripts/validate_features.py > logs/validation_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "✅ Validation restarted with PID: \$!"
sleep 2
tail -f logs/validation_*.log | grep --line-buffered -E '(⚠️|ERROR|train:|val:|SUMMARY|Validating)'
"@

Write-Host "📡 Connecting to remote server..." -ForegroundColor Yellow
ssh eoghan@192.168.200.206 $sshCommand
