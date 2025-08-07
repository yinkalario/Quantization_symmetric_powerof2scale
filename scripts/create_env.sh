#!/bin/bash

# Power-of-2 Symmetric Quantization Environment Setup Script
# Creates a conda environment and installs required packages

# Author: Yin Cao

set -e  # Exit on any error

ENV_NAME="QuantizationPo2"
PYTHON_VERSION="3.9"

echo "=== Power-of-2 Symmetric Quantization Environment Setup ==="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda not found. Please install Anaconda or Miniconda."
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found. Run from project root directory."
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing ${ENV_NAME} environment..."
    conda remove -n ${ENV_NAME} --all -y
fi

# Create conda environment
echo "Creating conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Install packages
echo "Installing required packages..."

# Initialize conda for bash (needed for conda activate)
eval "$(conda shell.bash hook)"

# Activate the environment
echo "Activating environment: ${ENV_NAME}"
conda activate ${ENV_NAME}

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing packages from requirements.txt..."
echo "This may take a few minutes. Installing packages one by one for better visibility:"
echo ""

# Read requirements.txt and install packages one by one
while IFS= read -r package || [ -n "$package" ]; do
    # Skip empty lines and comments
    if [[ -n "$package" && ! "$package" =~ ^[[:space:]]*# ]]; then
        echo "📦 Installing: $package"
        pip install "$package"
        echo "✅ Completed: $package"
        echo ""
    fi
done < requirements.txt

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Deactivate environment
conda deactivate

echo "🎉 All packages installed successfully!"

echo "✓ Environment setup complete!"
echo ""
echo "To activate: conda activate ${ENV_NAME}"
echo "To deactivate: conda deactivate"
echo ""
echo "To test the quantization, run:"
echo "  conda activate ${ENV_NAME}"
echo "  python test_quantization.py"
echo ""
echo "Usage examples:"
echo "  PTQ: python ptq_quantize.py --data_path ./data"
echo "  QAT: python qat_train.py --data_path ./data --epochs 10"
