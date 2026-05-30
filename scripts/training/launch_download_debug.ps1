#!/usr/bin/env pwsh
# Debug launcher for download

$logFile = "C:\Users\Eoghan Hynes\Wspc\concert_engagement\launch_debug.log"

"===========================================================" | Out-File $logFile
"Launch Debug Log - $(Get-Date)" | Out-File $logFile -Append
"===========================================================" | Out-File $logFile -Append
"" | Out-File $logFile -Append

try {
    "Creating SSH command..." | Out-File $logFile -Append
    
    $cmd = @"
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host 'Disengagement Actions Download' -ForegroundColor Green
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''
Write-Host 'Connecting to eoghan@192.168.200.206...' -ForegroundColor Yellow
Write-Host ''

ssh eoghan@192.168.200.206 "echo '=== Connection successful ==='; hostname; whoami; python3 --version; echo ''; echo 'Changing directory...'; cd ~/concert_engagement && echo 'Current dir:' && pwd && echo '' && echo 'Running download script...' && echo '' && python3 scripts/download_disengagement_actions.py"

Write-Host ''
Write-Host 'Press Enter to close...' -ForegroundColor Yellow
Read-Host
"@

    "SSH command created" | Out-File $logFile -Append
    "Command length: $($cmd.Length)" | Out-File $logFile -Append
    "" | Out-File $logFile -Append
    
    "Launching external window..." | Out-File $logFile -Append
    Start-Process pwsh -ArgumentList "-NoExit", "-Command", $cmd
    
    "Window launched successfully" | Out-File $logFile -Append
    
    Write-Host "✓ External window launched!" -ForegroundColor Green
    Write-Host "📝 Debug log: $logFile" -ForegroundColor Gray
    
} catch {
    "ERROR: $_" | Out-File $logFile -Append
    Write-Host "❌ Error: $_" -ForegroundColor Red
    Write-Host "📝 Check log: $logFile" -ForegroundColor Gray
}
