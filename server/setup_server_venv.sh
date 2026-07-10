#!/bin/bash
# Setup Python environment on remote server for AVA feature extraction

echo "=== Setting up Python environment ==="

# Create venv
echo "Creating virtual environment..."
python3 -m venv ~/concert_engagement/venv --without-pip

# Download and install pip
echo "Installing pip..."
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
~/concert_engagement/venv/bin/python3 /tmp/get-pip.py

# Install packages
echo "Installing Python packages (this will take a few minutes)..."
~/concert_engagement/venv/bin/pip install opencv-python mediapipe ultralytics torch torchvision numpy

echo ""
echo "=== Setup Complete ==="
echo "To use: ~/concert_engagement/venv/bin/python3 extract_ava_features.py"
