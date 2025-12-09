# download_assets.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import os


MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "ag_news"


MODEL_PATH = f"./offline_assets/{MODEL_NAME}"
DATASET_PATH = f"./offline_assets/{DATASET_NAME}"


print(f"Downloading model and tokenizer: {MODEL_NAME}...")

os.makedirs(MODEL_PATH, exist_ok=True)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(MODEL_PATH)


model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
model.save_pretrained(MODEL_PATH)

print(f"Model and tokenizer saved to: {MODEL_PATH}")


print(f"\nDownloading dataset: {DATASET_NAME}...")

os.makedirs(DATASET_PATH, exist_ok=True)


dataset = load_dataset(DATASET_NAME)
dataset.save_to_disk(DATASET_PATH)

print(f"Dataset saved to: {DATASET_PATH}")
print("\nPreparation complete. Copy the 'offline_assets' directory to your offline server.")