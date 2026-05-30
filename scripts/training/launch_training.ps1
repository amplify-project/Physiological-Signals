#!/usr/bin/env pwsh
# Launch training in external PowerShell window with visible output

Write-Host "Starting Kinetics-700 Transformer training on 192.168.200.206..." -ForegroundColor Cyan
Write-Host "This will open in a new PowerShell window showing live training output." -ForegroundColor Yellow
Write-Host ""

$command = @"
Write-Host "Connecting to server and starting training with nohup..." -ForegroundColor Cyan
ssh eoghan@192.168.200.206 'cd ~/concert_engagement && source venv/bin/activate && nohup python scripts/train_action_transformer_dataparallel.py --features-dir data/processed/features_kinetics700 --output-dir models/action_transformer_kinetics700 --batch-size 32 --epochs 100 --lr 1e-4 --sequence-length 300 > logs/training_kinetics_transformer_dataparallel.log 2>&1 &'
Write-Host ""
Write-Host "✅ Training started in background with nohup (survives VPN disconnects)" -ForegroundColor Green
Write-Host "📄 Log file: logs/training_kinetics_transformer_dataparallel.log" -ForegroundColor Cyan
Write-Host ""
Write-Host "Following live log output (Ctrl+C to stop watching, training continues)..." -ForegroundColor Yellow
Write-Host ""
Start-Sleep -Seconds 2
ssh eoghan@192.168.200.206 'tail -f ~/concert_engagement/logs/training_kinetics_transformer_dataparallel.log'
"@

Start-Process pwsh -ArgumentList "-NoExit", "-Command", $command

Write-Host "✅ Training launched in background with nohup!" -ForegroundColor Green
Write-Host "   (Survives SSH/VPN disconnects)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Monitor with: .\scripts\check_training.ps1" -ForegroundColor Cyan
Write-Host ""
