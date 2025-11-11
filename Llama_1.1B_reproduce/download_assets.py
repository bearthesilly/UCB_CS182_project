# download_assets.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# --- Configuration ---
OFFLINE_DIR = "./offline_assets"
# # 1. Model: Use the 1.1B TinyLlama model
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# # 2. Task A Dataset
# HOTPOT_DATASET_NAME = "hotpot_qa"
# HOTPOT_DATASET_CONFIG = "distractor"

# 3. Task B Dataset
MATH_DATASET_NAME = "qwedsacf/competition_math"

# # --- Create Directory ---
# if not os.path.exists(OFFLINE_DIR):
#     os.makedirs(OFFLINE_DIR)
#     print(f"Created directory: {OFFLINE_DIR}")

# # --- 1. Download Model and Tokenizer ---
# print(f"--- Downloading Model: {MODEL_NAME} ---")
# MODEL_PATH = os.path.join(OFFLINE_DIR, MODEL_NAME.replace("/", "_")) # Safer path name

# # Download and save tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# tokenizer.save_pretrained(MODEL_PATH)
# print(f"Tokenizer saved to {MODEL_PATH}")

# # Download and save model
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# model.save_pretrained(MODEL_PATH)
# print(f"Model saved to {MODEL_PATH}")


# # --- 2. Download Task A: HotpotQA ---
# print(f"\n--- Downloading Dataset: {HOTPOT_DATASET_NAME} ---")
# HOTPOT_PATH = os.path.join(OFFLINE_DIR, "hotpot_qa")

# hotpot_dataset = load_dataset(HOTPOT_DATASET_NAME, HOTPOT_DATASET_CONFIG)
# hotpot_dataset.save_to_disk(HOTPOT_PATH)
# print(f"HotpotQA dataset saved to {HOTPOT_PATH}")


# --- 3. Download Task B: MATH ---
print(f"\n--- Downloading Dataset: {MATH_DATASET_NAME} ---")
MATH_PATH = os.path.join(OFFLINE_DIR, "hendrycks_math")

math_dataset = load_dataset(MATH_DATASET_NAME)
math_dataset.save_to_disk(MATH_PATH)
print(f"MATH dataset saved to {MATH_PATH}")

print("\n--- All assets downloaded and saved successfully. ---")
print(f"Please transfer the entire '{OFFLINE_DIR}' directory to your offline server.")