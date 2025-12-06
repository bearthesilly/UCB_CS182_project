#!/usr/bin/env bash

# ========== CONFIG ==========
TARGET_PID=27278          # PID to wait for
CONDA_ENV=llm             # conda env name
PYTHON_SCRIPT="cs182_proj_python.py" # path to your python script
# ============================

echo "[INFO] Waiting for process $TARGET_PID to finish..."

# Loop until PID disappears
while kill -0 $TARGET_PID 2>/dev/null; do
    echo "[INFO] still waiting for process $TARGET_PID to finish..."
    sleep 1
done

echo "[INFO] Process $TARGET_PID finished. Launching Python script..."

# Initialize Conda
# This works for most installations
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate env
conda activate "$CONDA_ENV"

# Run your python script
python "$PYTHON_SCRIPT"

echo "[INFO] Script completed."