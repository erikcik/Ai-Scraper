#!/bin/bash

# Create and activate virtual environment
echo "Creating virtual environment..."
python -m venv env
source env/bin/activate

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention and dependencies
echo "Installing Flash Attention dependencies..."
pip install packaging
pip install ninja
pip install wheel
pip install --no-build-isolation flash-attn==2.7.4.post1

# Install Axolotl with Flash Attention support
echo "Installing Axolotl..."
pip install --no-build-isolation axolotl[flash-attn,deepspeed]

# Install other required packages
echo "Installing other dependencies..."
pip install transformers
pip install datasets
pip install tqdm
pip install argparse

# Run GPU check
echo "Running GPU check..."
python shadform_scripts/check_gpu.py

# Print completion message
echo "Setup completed! Virtual environment is active." 