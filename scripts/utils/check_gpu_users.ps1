# Show GPU processes with user information
$server = "eoghan@192.168.200.206"

Write-Host "=== GPU Processes by User ===" -ForegroundColor Cyan
Write-Host ""

# Show processes running on GPUs with user names
ssh $server "nvidia-smi dmon -s u"

Write-Host "`n=== Full GPU Status ===" -ForegroundColor Cyan
ssh $server "nvidia-smi pmon -s u"
