# download_assets.py
"""
Downloads required datasets and models for math vs coding catastrophic forgetting experiment.
"""
import os
from datasets import load_dataset
from transformers import AutoTokenizer

# Download math dataset (same as original)
OFFLINE_DIR = "./offline_assets"
math_dataset = load_dataset("qwedsacf/competition_math")
MATH_PATH = os.path.join(OFFLINE_DIR, "math")
math_dataset.save_to_disk(MATH_PATH)

# Download coding dataset (using smaller 'mbpp')
coding_dataset = load_dataset("mbpp")
CODING_PATH = os.path.join(OFFLINE_DIR, "mbpp")
coding_dataset.save_to_disk(CODING_PATH)

# Download TinyLlama model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Assets downloaded: math, coding datasets, TinyLlama model/tokenizer.")
