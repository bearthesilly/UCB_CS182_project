#!/bin/bash
set -e

echo "=== Installing Miniconda ==="
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -u -p $HOME/miniconda

# auto init conda for future logins
echo 'eval "$($HOME/miniconda/bin/conda shell.bash hook)"' >> ~/.bashrc
eval "$($HOME/miniconda/bin/conda shell.bash hook)"

echo "=== Accepting Anaconda Terms of Service ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "=== Creating conda env ==="
conda create -y -n llm python=3.12
conda activate llm

echo "=== Installing PyTorch (CUDA 12.1 wheel, compatible w/ CUDA 13 drivers) ==="
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing Hugging Face + dependencies ==="
pip install \
    transformers \
    datasets \
    peft \
    accelerate \
    bitsandbytes \
    sentencepiece \
    tqdm \
    matplotlib \
    pandas \
    scikit-learn

echo "=== DONE ==="
echo "Next SSH login will automatically load conda + llm env."
