#!/usr/bin/env pwsh
# Launch validation in external window

Write-Host "Starting feature validation on 192.168.200.206..." -ForegroundColor Cyan
Write-Host "This will scan ~71,000 .npz files and identify corrupted ones." -ForegroundColor Yellow
Write-Host ""

$command = @"
Write-Host "Connecting to server and starting validation..." -ForegroundColor Cyan
ssh -t eoghan@192.168.200.206 'cd ~/concert_engagement && source venv/bin/activate && python scripts/validate_features.py data/processed/features_kinetics700'
"@

Start-Process pwsh -ArgumentList "-NoExit", "-Command", $command

Write-Host "✅ Validation launched in external window!" -ForegroundColor Green
Write-Host ""
Write-Host "The window will show:" -ForegroundColor Cyan
Write-Host "  - Progress bar for train and val datasets" -ForegroundColor White
Write-Host "  - List of corrupted files found" -ForegroundColor White
Write-Host "  - Total count at the end" -ForegroundColor White
Write-Host "  - Prompt to delete corrupted files (type 'yes' to confirm)" -ForegroundColor White
Write-Host ""
