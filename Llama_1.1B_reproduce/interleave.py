"""
interleave_training.py

This script demonstrates an interleaved training strategy to mitigate
catastrophic forgetting. It trains a model on two tasks (MATH and HotpotQA)
by alternating batches from each task.
"""

import torch
import json
import os
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import warnings
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HOTPOT_DATASET_NAME = "hotpot_qa"
HOTPOT_DATASET_CONFIG = "distractor"
MATH_DATASET_NAME = "qwedsacf/competition_math"
RESULTS_DIR = "./drive/MyDrive/"

# --- VRAM-Saving Config ---
MAX_SEQ_LENGTH = 2048
PER_DEVICE_BS = 64 # This will be the batch size for *each* task's DataLoader
GRAD_ACC_STEPS = 1 # Kept for consistency, but manual loop handles 1 step at a time

# --- Experiment Config ---
N_TRAIN_EXAMPLES = 4000
N_VAL_EXAMPLES = 400
LEARNING_RATE = 7e-5       # Using the rate from the original Task A / Joint training
INTERLEAVE_EPOCHS = 4      # Doubled epochs (was 2 in baseline)
LOGGING_STEPS = 10         # Evaluate every 10 *batches* (e.g., 5 MATH, 5 HotpotQA)

# --- 2. Utility Functions (Data Formatting - Llama Chat Style) ---
def format_hotpot_qa(example):
    """Formats HotpotQA data into a Llama-chat-style prompt."""
    context = " ".join(["".join(s) for s in example["context"]["sentences"]])
    question = example["question"]
    answer = example["answer"]

    text = (
        f"<s>[INST] You are a helpful assistant. Use the following context to "
        f"answer the question. Context: {context}\n\nQuestion: {question} [/INST] "
        f"Answer: {answer}</s>"
    )
    return text

def format_math(example):
    """Formats MATH data into a Llama-chat-style prompt."""
    problem = example["problem"]
    solution = example["solution"]

    text = (
        f"<s>[INST] You are a math expert. Solve the following math problem. "
        f"Show your work.\nProblem: {problem} [/INST] "
        f"Solution: {solution}</s>"
    )
    return text

def filter_by_length(example, tokenizer, formatter):
    """
    Filters out examples that are too long.
    Returns True (keep) or False (discard).
    """
    text = formatter(example)
    tokenized = tokenizer(text, max_length=MAX_SEQ_LENGTH + 1, truncation=False, padding=False)
    return len(tokenized['input_ids']) <= MAX_SEQ_LENGTH

def preprocess(example, tokenizer, formatter):
    """
    Formats text, applies loss mask, and pads to max length.
    """
    text = formatter(example)
    tokenized = tokenizer(
        text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length", # Pad to max length
    )
    labels = tokenized["input_ids"].copy()
    inst_token_id = tokenizer.convert_tokens_to_ids("]")

    # Find the end of the prompt (the ']' token)
    split_point = -1
    for i in range(len(tokenized["input_ids"]) - 1, -1, -1):
        if tokenized["input_ids"][i] == inst_token_id:
            split_point = i + 1
            break

    if split_point == -1:
        # If prompt format is not found, discard the example
        return {}

    # Mask out the prompt tokens (set to -100)
    for i in range(split_point):
        labels[i] = -100

    tokenized["labels"] = labels
    return tokenized

# --- 3. Model Loading (Restructured) ---

def get_model_and_tokenizer_base():
    """
    Loads the base FP16 TinyLlama model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16, # Load in FP16 to save RAM
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Enable gradient checkpointing on the base model
    model.gradient_checkpointing_enable()

    return model, tokenizer

def get_lora_config():
    """
    Defines the LoRA configuration.
    """
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

def manual_evaluate(model, dataloader, device):
    """
    Manually runs evaluation on a given dataloader.
    """
    model.eval()  # <-- Set to evaluation mode
    total_loss = 0
    total_steps = 0
    with torch.no_grad(): # <-- Disable gradient calculation
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move batch to the same device as the model
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}

            outputs = model(**batch)
            loss = outputs.loss

            total_loss += loss.item()
            total_steps += 1

    model.train() # <-- [Important] Set model back to training mode
    if total_steps == 0:
        return 0.0
    return total_loss / total_steps

# --- 4. Main Experiment Logic ---
def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print(f"--- Loading Base Model & Tokenizer ---")
    base_model, tokenizer = get_model_and_tokenizer_base()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Load and Process Datasets ---
    print(f"\n--- Loading and Preprocessing Datasets (This may take a while) ---")

    # Task B: HotpotQA
    raw_hotpot = load_dataset(HOTPOT_DATASET_NAME, HOTPOT_DATASET_CONFIG)
    hotpot_train = raw_hotpot["train"].shuffle(seed=42).select(range(N_TRAIN_EXAMPLES))
    hotpot_val = raw_hotpot["validation"].shuffle(seed=42).select(range(N_VAL_EXAMPLES))

    print(f"Tokenizing and filtering HotpotQA...")
    hotpot_train_tokenized = hotpot_train.filter(
        lambda x: filter_by_length(x, tokenizer, format_hotpot_qa),
        batched=False,
    ).map(
        lambda x: preprocess(x, tokenizer, format_hotpot_qa),
        batched=False,
    ).filter(lambda example: len(example) > 0)

    hotpot_val_tokenized = hotpot_val.filter(
        lambda x: filter_by_length(x, tokenizer, format_hotpot_qa),
        batched=False,
    ).map(
        lambda x: preprocess(x, tokenizer, format_hotpot_qa),
        batched=False,
    ).filter(lambda example: len(example) > 0)

    print(f"HotpotQA: {len(hotpot_train_tokenized)} train, {len(hotpot_val_tokenized)} val (after filtering)")

    # Task A: MATH
    raw_math = load_dataset(MATH_DATASET_NAME)
    total_math_samples_needed = N_TRAIN_EXAMPLES + N_VAL_EXAMPLES
    math_subset = raw_math["train"].shuffle(seed=42).select(range(total_math_samples_needed))
    val_size_fraction = N_VAL_EXAMPLES / total_math_samples_needed
    math_splits = math_subset.train_test_split(test_size=val_size_fraction, seed=42)
    math_train = math_splits["train"]
    math_val = math_splits["test"]

    print(f"Tokenizing and filtering MATH...")
    math_train_tokenized = math_train.filter(
        lambda x: filter_by_length(x, tokenizer, format_math),
        batched=False,
    ).map(
        lambda x: preprocess(x, tokenizer, format_math),
        batched=False,
    ).filter(lambda example: len(example) > 0)

    math_val_tokenized = math_val.filter(
        lambda x: filter_by_length(x, tokenizer, format_math),
        batched=False,
    ).map(
        lambda x: preprocess(x, tokenizer, format_math),
        batched=False,
    ).filter(lambda example: len(example) > 0)

    print(f"MATH: {len(math_train_tokenized)} train, {len(math_val_tokenized)} val (after filtering)")
    
    # --- Remove token_type_ids as it's not needed by Llama ---
    # This can also help with potential key mismatches
    hotpot_train_tokenized = hotpot_train_tokenized.remove_columns(["token_type_ids"], errors='ignore')
    hotpot_val_tokenized = hotpot_val_tokenized.remove_columns(["token_type_ids"], errors='ignore')
    math_train_tokenized = math_train_tokenized.remove_columns(["token_type_ids"], errors='ignore')
    math_val_tokenized = math_val_tokenized.remove_columns(["token_type_ids"], errors='ignore')


    # --- Setup for Manual Interleaved Training ---
    print(f"\n--- Setting up for Interleaved Training ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get LoRA model
    lora_config = get_lora_config()
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    # model.to(device) # device_map="auto" already handled this
    model.train() # Set to training mode

    # Create DataLoaders
    math_train_loader = DataLoader(
        math_train_tokenized,
        batch_size=PER_DEVICE_BS,
        collate_fn=data_collator,
        shuffle=True
    )
    hotpot_train_loader = DataLoader(
        hotpot_train_tokenized,
        batch_size=PER_DEVICE_BS,
        collate_fn=data_collator,
        shuffle=True
    )
    math_val_loader = DataLoader(
        math_val_tokenized,
        batch_size=PER_DEVICE_BS,
        collate_fn=data_collator
    )
    hotpot_val_loader = DataLoader(
        hotpot_val_tokenized,
        batch_size=PER_DEVICE_BS,
        collate_fn=data_collator
    )
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # History tracking
    history = {"steps": [], "hotpot_loss": [], "math_loss": []}
    global_step = 0

    # --- Initial Evaluation (Step 0) ---
    print("\n--- Running initial evaluation (Step 0) ---")
    hotpot_loss_0 = manual_evaluate(model, hotpot_val_loader, device)
    math_loss_0 = manual_evaluate(model, math_val_loader, device)
    
    history["steps"].append(0)
    history["hotpot_loss"].append(hotpot_loss_0)
    history["math_loss"].append(math_loss_0)
    print(f"  > Step 0 - HotpotQA Val Loss: {hotpot_loss_0:.4f}")
    print(f"  > Step 0 - MATH Val Loss: {math_loss_0:.4f}")

    # --- Start Interleaved Training Loop ---
    print(f"\n--- Starting Interleaved Training for {INTERLEAVE_EPOCHS} epochs ---")
    
    for epoch in range(INTERLEAVE_EPOCHS):
        print(f"\n--- Starting Epoch {epoch + 1}/{INTERLEAVE_EPOCHS} ---")
        model.train()
        
        # Create iterators for both datasets
        math_iter = iter(math_train_loader)
        hotpot_iter = iter(hotpot_train_loader)
        
        # We loop for the number of batches in the *smaller* dataset
        num_batches = min(len(math_train_loader), len(hotpot_train_loader))
        
        for i in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}"):
            
            # --- Task A (MATH) Batch ---
            try:
                math_batch = next(math_iter)
            except StopIteration:
                # Should not happen with this loop structure, but as a safeguard
                math_iter = iter(math_train_loader)
                math_batch = next(math_iter)
            
            math_batch = {k: v.to(device) for k, v in math_batch.items() if k in ["input_ids", "attention_mask", "labels"]}
            
            outputs_math = model(**math_batch)
            loss_math = outputs_math.loss
            
            loss_math.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1 # Increment global step after *each* batch
            
            # --- Log if needed ---
            if global_step % LOGGING_STEPS == 0:
                print(f"\n--- Evaluation at Step {global_step} (after MATH batch) ---")
                hotpot_loss = manual_evaluate(model, hotpot_val_loader, device)
                math_loss = manual_evaluate(model, math_val_loader, device)
                history["steps"].append(global_step)
                history["hotpot_loss"].append(hotpot_loss)
                history["math_loss"].append(math_loss)
                print(f"  > Step {global_step} - HotpotQA Val Loss: {hotpot_loss:.4f}")
                print(f"  > Step {global_step} - MATH Val Loss: {math_loss:.4f}")
                model.train() # Ensure model is back in train mode

            # --- Task B (HotpotQA) Batch ---
            try:
                hotpot_batch = next(hotpot_iter)
            except StopIteration:
                hotpot_iter = iter(hotpot_train_loader)
                hotpot_batch = next(hotpot_iter)

            hotpot_batch = {k: v.to(device) for k, v in hotpot_batch.items() if k in ["input_ids", "attention_mask", "labels"]}
            
            outputs_hotpot = model(**hotpot_batch)
            loss_hotpot = outputs_hotpot.loss
            
            loss_hotpot.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1 # Increment global step again
            
            # --- Log if needed ---
            if global_step % LOGGING_STEPS == 0:
                print(f"\n--- Evaluation at Step {global_step} (after HotpotQA batch) ---")
                hotpot_loss = manual_evaluate(model, hotpot_val_loader, device)
                math_loss = manual_evaluate(model, math_val_loader, device)
                history["steps"].append(global_step)
                history["hotpot_loss"].append(hotpot_loss)
                history["math_loss"].append(math_loss)
                print(f"  > Step {global_step} - HotpotQA Val Loss: {hotpot_loss:.4f}")
                print(f"  > Step {global_step} - MATH Val Loss: {math_loss:.4f}")
                model.train() # Ensure model is back in train mode

    # --- 5. Plot Results ---
    print("\n--- Training Complete. Saving History Data and Generating Plot ---")

    # --- Save history data to JSON ---
    history_filename = os.path.join(RESULTS_DIR, "interleave_history_MATH_and_HotpotQA_fp32.json")
    try:
        with open(history_filename, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"History data saved to {history_filename}")
    except Exception as e:
        print(f"Error saving history to JSON: {e}")
    # --- [END] ---

    plt.figure(figsize=(12, 6))
    plt.plot(history["steps"], history["hotpot_loss"], 'o-', label="Task B (HotpotQA) Loss", color="blue")
    plt.plot(history["steps"], history["math_loss"], 'o-', label="Task A (MATH) Loss", color="red")

    plt.title(f"Interleaved Training: MATH + HotpotQA (Model: {MODEL_NAME} FP32 LoRA)")
    plt.xlabel(f"Training Steps (Total Batches) (Total Epochs: {INTERLEAVE_EPOCHS})")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()

    plot_filename = os.path.join(RESULTS_DIR, "interleave_curve_MATH_and_HotpotQA_fp32.png")
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    try:
        from google.colab import files
        plt.show()
    except ImportError:
        print("Not in Colab, plot saved to file.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: This experiment requires a GPU. Check Colab runtime type.")
    else:
        print(f"INFO: Running on GPU. VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        if torch.cuda.get_device_properties(0).total_memory / 1e9 < 11:
            print("WARNING: VRAM is less than 11GB. You may hit OOM errors. Try lowering MAX_SEQ_LENGTH.")
    main()