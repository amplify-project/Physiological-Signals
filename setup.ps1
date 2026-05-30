# =============================================================================
# Concert Engagement Analysis - Windows Setup Script
# =============================================================================
# This script sets up the development environment with proper CUDA support.
#
# Usage: .\setup.ps1
# =============================================================================

param(
    [switch]$CPU,           # Force CPU-only installation
    [switch]$SkipVenv,      # Skip virtual environment creation
    [switch]$Help
)

$FaceWeightsUrl = "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt"
$FaceWeightsPath = "models\20180402-114759-vggface2.pt"

if ($Help) {
    Write-Host @"
Concert Engagement Analysis - Setup Script

Usage: .\setup.ps1 [options]

Options:
    -CPU        Force CPU-only PyTorch installation (no CUDA)
    -SkipVenv   Skip virtual environment creation (use existing)
    -Help       Show this help message

Examples:
    .\setup.ps1              # Full setup with CUDA auto-detection
    .\setup.ps1 -CPU         # Setup with CPU-only PyTorch
    .\setup.ps1 -SkipVenv    # Only install packages (venv already exists)
"@
    exit 0
}

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " Concert Engagement Analysis - Setup" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Please install Python 3.8-3.12" -ForegroundColor Red
    exit 1
}
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# Block Microsoft Store Python (its sandbox breaks camera/hardware access)
$pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
if ($pythonPath -and $pythonPath -match 'WindowsApps') {
    Write-Host ""
    Write-Host "ERROR: Microsoft Store Python detected ($pythonPath)" -ForegroundColor Red
    Write-Host "       Its sandbox restricts camera and hardware access." -ForegroundColor Red
    Write-Host "       Install Python from https://www.python.org/downloads/ instead." -ForegroundColor Yellow
    exit 1
}

# Create virtual environment
if (-not $SkipVenv) {
    Write-Host ""
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    
    if (Test-Path ".venv") {
        Write-Host "Removing existing .venv..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force ".venv"
    }
    
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual environment created." -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Detect CUDA availability
$cudaAvailable = $false
if (-not $CPU) {
    Write-Host ""
    Write-Host "Checking for NVIDIA GPU..." -ForegroundColor Yellow
    
    try {
        $nvidiaSmi = nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Found GPU: $nvidiaSmi" -ForegroundColor Green
            $cudaAvailable = $true
        }
    } catch {
        Write-Host "No NVIDIA GPU detected." -ForegroundColor Yellow
    }
}

# Install PyTorch
Write-Host ""
Write-Host "Installing PyTorch..." -ForegroundColor Yellow

if ($CPU) {
    Write-Host "Installing CPU-only version (forced)..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
} elseif ($cudaAvailable) {
    Write-Host "Installing with CUDA 12.4 support..." -ForegroundColor Green
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Host "Installing CPU-only version (no GPU detected)..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install PyTorch" -ForegroundColor Red
    exit 1
}

# Verify PyTorch installation
Write-Host ""
Write-Host "Verifying PyTorch installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Install remaining requirements
Write-Host ""
Write-Host "Installing remaining dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install requirements" -ForegroundColor Red
    exit 1
}

# Install FaceNet without letting its outdated metadata downgrade PyTorch
Write-Host ""
Write-Host "Installing facenet-pytorch..." -ForegroundColor Yellow
pip install facenet-pytorch==2.6.0 --no-deps

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install facenet-pytorch" -ForegroundColor Red
    exit 1
}

# Ensure FaceNet weights are available before runtime
Write-Host ""
Write-Host "Ensuring face ID weights are available locally..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "models" | Out-Null

if (Test-Path $FaceWeightsPath) {
    Write-Host "Face ID weights already present." -ForegroundColor Green
} else {
    try {
        Invoke-WebRequest -Uri $FaceWeightsUrl -OutFile $FaceWeightsPath
    } catch {
        Write-Host "ERROR: Could not download the Face ID weights." -ForegroundColor Red
        Write-Host "Download this file manually and place it here:" -ForegroundColor Yellow
        Write-Host "  $FaceWeightsPath" -ForegroundColor White
        Write-Host "URL:" -ForegroundColor Yellow
        Write-Host "  $FaceWeightsUrl" -ForegroundColor White
        exit 1
    }
}

# Final verification
Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment in future sessions:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To run the engagement system:" -ForegroundColor Yellow
Write-Host "  1. Start Redis: .\redis\redis-server.exe" -ForegroundColor White
Write-Host "  2. Run inference: python scripts/live_multiperson_binary_v2.py" -ForegroundColor White
Write-Host ""

# Show final status
python -c @"
import torch
import platform

print('Environment Summary:')
print(f'  Platform: {platform.system()} ({platform.machine()})')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"@
