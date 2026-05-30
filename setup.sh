#!/bin/bash
# =============================================================================
# Concert Engagement Analysis - macOS/Linux Setup Script
# =============================================================================
# This script sets up the development environment with proper GPU support.
#
# Usage: ./setup.sh [options]
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default options
CPU_ONLY=false
SKIP_VENV=false
FACE_WEIGHTS_URL="https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt"
FACE_WEIGHTS_PATH="models/20180402-114759-vggface2.pt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            CPU_ONLY=true
            shift
            ;;
        --skip-venv)
            SKIP_VENV=true
            shift
            ;;
        --help|-h)
            echo "Concert Engagement Analysis - Setup Script"
            echo ""
            echo "Usage: ./setup.sh [options]"
            echo ""
            echo "Options:"
            echo "    --cpu         Force CPU-only PyTorch installation"
            echo "    --skip-venv   Skip virtual environment creation"
            echo "    --help, -h    Show this help message"
            echo ""
            echo "Examples:"
            echo "    ./setup.sh              # Full setup with GPU auto-detection"
            echo "    ./setup.sh --cpu        # Setup with CPU-only PyTorch"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}=============================================${NC}"
echo -e "${CYAN} Concert Engagement Analysis - Setup${NC}"
echo -e "${CYAN}=============================================${NC}"
echo ""

# Detect OS
OS_TYPE=$(uname -s)
ARCH=$(uname -m)

echo -e "${GREEN}Detected: $OS_TYPE ($ARCH)${NC}"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python3 not found. Please install Python 3.8-3.11${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}Found: $PYTHON_VERSION${NC}"

# Create virtual environment
if [ "$SKIP_VENV" = false ]; then
    echo ""
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    
    if [ -d ".venv" ]; then
        echo -e "${YELLOW}Removing existing .venv...${NC}"
        rm -rf .venv
    fi
    
    python3 -m venv .venv
    echo -e "${GREEN}Virtual environment created.${NC}"
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip
echo ""
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Determine PyTorch installation
echo ""
echo -e "${YELLOW}Detecting GPU support...${NC}"

if [ "$CPU_ONLY" = true ]; then
    echo -e "${YELLOW}Installing CPU-only version (forced)...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
elif [ "$OS_TYPE" = "Darwin" ]; then
    # macOS
    if [ "$ARCH" = "arm64" ]; then
        echo -e "${GREEN}Apple Silicon detected - MPS acceleration available${NC}"
        # Standard PyTorch includes MPS support for Apple Silicon
        pip install torch torchvision torchaudio
    else
        echo -e "${YELLOW}Intel Mac detected - CPU only${NC}"
        pip install torch torchvision torchaudio
    fi
elif [ "$OS_TYPE" = "Linux" ]; then
    # Linux - check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "")
        if [ -n "$GPU_NAME" ]; then
            echo -e "${GREEN}Found GPU: $GPU_NAME${NC}"
            echo -e "${GREEN}Installing with CUDA 12.4 support...${NC}"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        else
            echo -e "${YELLOW}No NVIDIA GPU detected - installing CPU version${NC}"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    else
        echo -e "${YELLOW}nvidia-smi not found - installing CPU version${NC}"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo -e "${YELLOW}Unknown OS - installing CPU version${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Verify PyTorch installation
echo ""
echo -e "${YELLOW}Verifying PyTorch installation...${NC}"
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"

# Install remaining requirements
echo ""
echo -e "${YELLOW}Installing remaining dependencies...${NC}"
pip install -r requirements.txt

# Install FaceNet without letting its outdated metadata downgrade PyTorch
echo ""
echo -e "${YELLOW}Installing facenet-pytorch...${NC}"
pip install facenet-pytorch==2.6.0 --no-deps

# Ensure FaceNet weights are available before runtime
echo ""
echo -e "${YELLOW}Ensuring face ID weights are available locally...${NC}"
mkdir -p models
if [ -f "$FACE_WEIGHTS_PATH" ]; then
    echo -e "${GREEN}Face ID weights already present.${NC}"
else
    python3 - << EOF
from pathlib import Path
from urllib.request import urlretrieve

url = "$FACE_WEIGHTS_URL"
path = Path("$FACE_WEIGHTS_PATH")
try:
    urlretrieve(url, path)
except Exception as exc:
    print(f"ERROR: Could not download the Face ID weights: {exc}")
    print("Download this file manually and place it here:")
    print(f"  {path}")
    print("URL:")
    print(f"  {url}")
    raise SystemExit(1)
EOF
fi

# Final verification
echo ""
echo -e "${CYAN}=============================================${NC}"
echo -e "${GREEN} Setup Complete!${NC}"
echo -e "${CYAN}=============================================${NC}"
echo ""
echo -e "${YELLOW}To activate the environment in future sessions:${NC}"
echo "  source .venv/bin/activate"
echo ""
echo -e "${YELLOW}To run the engagement system:${NC}"
echo "  1. Start Redis: redis-server (install via brew/apt if needed)"
echo "  2. Run inference: python scripts/live_multiperson_binary_v2.py"
echo ""

# Show final status
python3 << 'EOF'
import torch
import platform

print('Environment Summary:')
print(f'  Platform: {platform.system()} ({platform.machine()})')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if hasattr(torch.backends, 'mps'):
    print(f'  MPS: {torch.backends.mps.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
EOF
