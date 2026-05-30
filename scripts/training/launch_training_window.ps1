# Launch training in external window for live monitoring
Start-Process pwsh -ArgumentList "-NoExit", "-Command", @"
ssh eoghan@192.168.200.206 'cd ~/concert_engagement && source venv/bin/activate && python scripts/train_action_transformer_dataparallel.py --features-dir data/processed/features_kinetics700 --batch-size 32 --epochs 50 --lr 0.0001'
"@
