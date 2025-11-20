# download_assets.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# --- Configuration ---
OFFLINE_DIR = "./offline_assets"
# 1. Model: Use the 1.1B TinyLlama model
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 2. Task A Dataset (Previously Task B, now Task A is Math, Task B is Code)
# Note: In the experiment logic, Task A is MATH. Here we just download assets.

# Task B Dataset: CodeParrot
CODE_DATASET_NAME = "codeparrot/codeparrot-clean"
CODE_DATASET_CONFIG = None

# Task A Dataset: MATH
MATH_DATASET_NAME = "qwedsacf/competition_math"

# # --- Create Directory ---
if not os.path.exists(OFFLINE_DIR):
    os.makedirs(OFFLINE_DIR)
    print(f"Created directory: {OFFLINE_DIR}")

# --- 1. Download Model and Tokenizer ---
print(f"--- Downloading Model: {MODEL_NAME} ---")
MODEL_PATH = os.path.join(OFFLINE_DIR, MODEL_NAME.replace("/", "_")) # Safer path name

# Download and save tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(MODEL_PATH)
print(f"Tokenizer saved to {MODEL_PATH}")

# Download and save model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.save_pretrained(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")


# --- 2. Download Task B: CodeParrot ---
print(f"\n--- Downloading Dataset: {CODE_DATASET_NAME} ---")
CODE_PATH = os.path.join(OFFLINE_DIR, "codeparrot")

# We use a subset to avoid downloading TBs of data
code_dataset = load_dataset(CODE_DATASET_NAME, CODE_DATASET_CONFIG, split="train[:20000]", trust_remote_code=True)
code_dataset.save_to_disk(CODE_PATH)
print(f"CodeParrot dataset (subset) saved to {CODE_PATH}")


# --- 3. Download Task A: MATH ---
print(f"\n--- Downloading Dataset: {MATH_DATASET_NAME} ---")
MATH_PATH = os.path.join(OFFLINE_DIR, "hendrycks_math")

math_dataset = load_dataset(MATH_DATASET_NAME)
math_dataset.save_to_disk(MATH_PATH)
print(f"MATH dataset saved to {MATH_PATH}")

print("\n--- All assets downloaded and saved successfully. ---")
print(f"Please transfer the entire '{OFFLINE_DIR}' directory to your offline server.")