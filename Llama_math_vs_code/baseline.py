# baseline.py
"""
Basic continual learning experiment: Train TinyLlama on math, then coding dataset, and measure catastrophic forgetting.
"""
import torch
import json
import os
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk, concatenate_datasets
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- Prompt Formatting Functions ---
def format_math(example):
    problem = example["problem"]
    solution = example.get("solution", "")
    return (
        f"<s>[INST] You are a math expert. Solve the following math problem. "
        f"Show your work.\nProblem: {problem} [/INST] "
        f"Solution: {solution}</s>"
    )

def format_code(example):
    prompt = example["text"]
    code = example.get("code", "")
    return (
        f"<s>[INST] You are a coding expert. Write code for the following prompt.\nPrompt: {prompt} [/INST] "
        f"Code: {code}</s>"
    )

# --- Preprocessing ---
def preprocess(example, tokenizer, formatter, max_length=1024):
    text = formatter(example)
    tokenized = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    labels = tokenized["input_ids"].copy()
    # Mask out prompt tokens (set to -100)
    inst_token_id = tokenizer.convert_tokens_to_ids("]")
    split_point = -1
    for i in range(len(tokenized["input_ids"]) - 1, -1, -1):
        if tokenized["input_ids"][i] == inst_token_id:
            split_point = i + 1
            break
    if split_point == -1:
        return {}
    for i in range(split_point):
        labels[i] = -100
    tokenized["labels"] = labels
    return tokenized

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MATH_PATH = "./offline_assets/math"
CODING_PATH = "./offline_assets/mbpp"
RESULTS_DIR = "./result_json/"
EPOCHS_MATH = 2
EPOCHS_CODE = 2
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)


# --- Load Datasets from disk ---
math_dataset = load_from_disk(MATH_PATH)['train']
coding_dataset = load_from_disk(CODING_PATH)['train']


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# Tokenize and preprocess with formatting
def preprocess_batch_math(batch):
    problems = batch["problem"]
    solutions = batch.get("solution", [""] * len(problems))
    outputs = [preprocess({"problem": p, "solution": s}, tokenizer, format_math, max_length=512)
               for p, s in zip(problems, solutions)]
    # Convert list of dicts to dict of lists
    if not outputs:
        return {}
    return {k: [out[k] for out in outputs] for k in outputs[0]}

def preprocess_batch_code(batch):
    texts = batch["text"]
    codes = batch.get("code", [""] * len(texts))
    outputs = [preprocess({"text": t, "code": c}, tokenizer, format_code, max_length=512)
               for t, c in zip(texts, codes)]
    if not outputs:
        return {}
    return {k: [out[k] for out in outputs] for k in outputs[0]}

math_tokenized = math_dataset.map(preprocess_batch_math, batched=True, remove_columns=math_dataset.column_names)
coding_tokenized = coding_dataset.map(preprocess_batch_code, batched=True, remove_columns=coding_dataset.column_names)


# --- Model + LoRA ---
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
base_model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# --- Training Arguments ---

training_args = TrainingArguments(
    output_dir="./outputs/",
    num_train_epochs=EPOCHS_MATH,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=500,
    logging_steps=100,
    report_to=[],
    gradient_checkpointing=True,
)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# --- Train on Math ---
trainer_math = Trainer(
    model=model,
    args=training_args,
    train_dataset=math_tokenized,
    data_collator=data_collator,
)
trainer_math.train()

# --- Evaluate on Math (after math) ---
math_loss_before = trainer_math.evaluate(math_tokenized)["eval_loss"]

# --- Train on Coding ---
training_args.num_train_epochs = EPOCHS_CODE
trainer_code = Trainer(
    model=model,
    args=training_args,
    train_dataset=coding_tokenized,
    data_collator=data_collator,
)
trainer_code.train()

# --- Evaluate on Math (after coding) ---
math_loss_after = trainer_code.evaluate(math_tokenized)["eval_loss"]
# --- Evaluate on Coding ---
coding_loss = trainer_code.evaluate(coding_tokenized)["eval_loss"]

# --- Save Results ---
results = {
    "math_loss_before": math_loss_before,
    "math_loss_after": math_loss_after,
    "coding_loss": coding_loss,
}
with open(os.path.join(RESULTS_DIR, "forgetting_history_math_vs_code.json"), "w") as f:
    json.dump(results, f)
print("Results saved to result_json/forgetting_history_math_vs_code.json")
