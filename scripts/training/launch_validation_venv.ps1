# Launch validation with venv in external window
Start-Process pwsh -ArgumentList "-NoExit", "-Command", @"
ssh eoghan@192.168.200.206 'cd ~/concert_engagement && source venv/bin/activate && python scripts/validate_features.py'
"@
