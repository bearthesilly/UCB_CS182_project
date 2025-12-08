import torch
import json
import os
import argparse
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BitsAndBytesConfig, # <-- 已移除
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, concatenate_datasets
from transformers.trainer_callback import TrainerCallback
# 【已修复】导入 PeftModel
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import warnings
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Utility Function for Unique Filenames ---
def get_unique_filename(base_path, base_name, extension=""):
    """
    Generate a unique filename by appending a number if the file already exists.

    Args:
        base_path: Directory where the file will be saved
        base_name: Base name of the file (without extension)
        extension: File extension (e.g., ".json", ".png")

    Returns:
        Unique filepath that doesn't exist yet
    """
    # Ensure extension starts with a dot if provided
    if extension and not extension.startswith("."):
        extension = "." + extension

    # Try the base name first
    filepath = os.path.join(base_path, f"{base_name}{extension}")

    # If it doesn't exist, return it
    if not os.path.exists(filepath):
        return filepath

    # Otherwise, append numbers until we find a unique name
    counter = 1
    while True:
        filepath = os.path.join(base_path, f"{base_name}_{counter}{extension}")
        if not os.path.exists(filepath):
            return filepath
        counter += 1

# --- 1. Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HOTPOT_DATASET_NAME = "hotpot_qa"
HOTPOT_DATASET_CONFIG = "distractor"
MATH_DATASET_NAME = "qwedsacf/competition_math"
RESULTS_DIR = "./drive/MyDrive/"

# 【检查点路径 1 - 对照组】
JOINT_ADAPTER_PATH = os.path.join(RESULTS_DIR, "joint_adapter_llama_fp32")

# 【检查点路径 2 - 实验组 Phase 1 (MATH)】
TASK_A_ADAPTER_PATH = os.path.join(RESULTS_DIR, "math_adapter_llama_fp32")


# --- VRAM-Saving Config ---
MAX_SEQ_LENGTH = 2048
# 【BUG 修复】降低 BS 以适应 FP32
PER_DEVICE_BS = 64
GRAD_ACC_STEPS = 1 # (有效批量大小仍然是 8 * 4 = 32)

# --- Experiment Config ---
N_TRAIN_EXAMPLES = 4000
N_VAL_EXAMPLES = 400
JOINT_EPOCHS = 2
TASK_A_EPOCHS = 2 # For Math
TASK_B_EPOCHS = 2 # For HotpotQA

# --- Data Mixing Config ---
MIX_PERCENTAGE = 0.1  # 10% of Phase A data to mix into Phase B
# SELECTION_METHOD will be passed as a command-line argument

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
    只检查长度。返回 True (保留) 或 False (丢弃)。
    """
    text = formatter(example)
    tokenized = tokenizer(text, max_length=MAX_SEQ_LENGTH + 1, truncation=False, padding=False)
    return len(tokenized['input_ids']) <= MAX_SEQ_LENGTH

# 【BUG 修复】这是修复了 1.7 Loss 问题 和 ValueError 的 Preprocess 函数
def preprocess(example, tokenizer, formatter):
    """
    【已修正】
    格式化文本，应用损失掩码，并填充到最大长度。
    """
    text = formatter(example)
    tokenized = tokenizer(
        text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length", # 修复 ValueError
    )
    labels = tokenized["input_ids"].copy()
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

# --- 3. Model Loading (【重构】) ---

def get_model_and_tokenizer_base():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 在基础模型上启用梯度检查点
    model.gradient_checkpointing_enable()

    return model, tokenizer

def get_lora_config():
    """
    只定义 LoRA 配置。
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
    在给定 dataloader 上手动运行评估。
    """
    model.eval()  # <--- 设置为评估模式
    total_loss = 0
    total_steps = 0
    with torch.no_grad(): # <--- 禁用梯度计算
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # 将批次移动到模型所在的设备
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}

            outputs = model(**batch)
            loss = outputs.loss

            total_loss += loss.item()
            total_steps += 1

    model.train() # <--- 【重要】将模型设置回训练模式
    return total_loss / total_steps

# --- Data Selection Functions for Mixing ---

def select_random(dataset, num_samples, **kwargs):
    """
    Randomly select samples from the dataset.

    Args:
        dataset: The tokenized dataset to select from
        num_samples: Number of samples to select

    Returns:
        Selected subset of the dataset
    """
    indices = list(range(len(dataset)))
    import random
    random.seed(42)
    selected_indices = random.sample(indices, min(num_samples, len(dataset)))
    return dataset.select(selected_indices)

def select_by_difficulty(dataset, num_samples, raw_dataset=None, reverse=True, **kwargs):
    """
    Select samples by difficulty level (highest difficulty first).
    Assumes the raw dataset has a 'level' field.

    Args:
        dataset: The tokenized dataset to select from
        num_samples: Number of samples to select
        raw_dataset: The raw (non-tokenized) dataset with 'level' field

    Returns:
        Selected subset of the dataset
    """
    if raw_dataset is None or 'level' not in raw_dataset.column_names:
        print("WARNING: 'level' field not found in dataset. Falling back to random selection.")
        return select_random(dataset, num_samples)

    # FIXED: Only use indices that exist in the tokenized dataset
    # Create a list of (index, level) tuples for only valid indices
    indexed_levels = [(i, raw_dataset[i]['level']) for i in range(len(dataset))]

    # Sort by level in descending order (highest difficulty first)
    # Level is like "Level 1", "Level 2", etc., so we extract the number
    def extract_level(level_str):
        try:
            return int(level_str.split()[-1])
        except:
            return 0

    indexed_levels.sort(key=lambda x: extract_level(x[1]), reverse=reverse)

    # Take top num_samples indices
    selected_indices = [idx for idx, _ in indexed_levels[:num_samples]]

    values = dataset.select(selected_indices)

    return values

def select_by_loss(dataset, num_samples, model=None, tokenizer=None, reverse=True, cache_dir=None, **kwargs):
    """
    Select samples by loss (highest or lowest loss first).

    Args:
        dataset: The tokenized dataset to select from
        num_samples: Number of samples to select
        model: The trained model to compute losses
        tokenizer: The tokenizer
        reverse: If True, select highest loss first (descending). If False, select lowest loss first (ascending).
        cache_dir: Directory to cache computed losses

    Returns:
        Selected subset of the dataset
    """
    if model is None:
        print("WARNING: Model not provided for loss-based selection. Falling back to random selection.")
        return select_random(dataset, num_samples)

    # Generate cache filename based on dataset size and model
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_filename = f"math_losses_{len(dataset)}_samples.json"
        cache_path = os.path.join(cache_dir, cache_filename)

        # Try to load cached losses
        if os.path.exists(cache_path):
            print(f"Loading cached losses from {cache_path}...")
            try:
                with open(cache_path, 'r') as f:
                    losses = json.load(f)
                print(f"Successfully loaded {len(losses)} cached losses")
            except Exception as e:
                print(f"Error loading cached losses: {e}. Recomputing...")
                losses = None
        else:
            losses = None
    else:
        losses = None

    # Compute losses if not cached
    if losses is None:
        print(f"Computing losses for {len(dataset)} samples (this may take a while)...")

        # Only keep the tokenized fields for the DataLoader
        dataset_for_loss = dataset.remove_columns([col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]])

        # Create a DataLoader for the dataset with smaller batch size to avoid OOM
        # Use batch size of 4 for loss computation (much smaller than training batch size)
        loss_batch_size = 8
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        dataloader = DataLoader(
            dataset_for_loss,
            batch_size=loss_batch_size,
            collate_fn=data_collator,
            shuffle=False
        )
        print(f"Using batch size of {loss_batch_size} for loss computation to avoid OOM")

        device = next(model.parameters()).device
        model.eval()

        losses = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing losses for selection"):
                batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}

                outputs = model(**batch)

                # Get per-sample losses
                # outputs.loss is the mean, but we need individual losses
                logits = outputs.logits
                labels = batch["labels"]

                # Compute cross-entropy loss per sample
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Flatten for loss computation
                loss_per_token = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # Reshape back to (batch_size, seq_len - 1)
                loss_per_token = loss_per_token.view(shift_labels.size())

                # Average loss per sample (ignoring -100 labels)
                for i in range(loss_per_token.size(0)):
                    valid_mask = shift_labels[i] != -100
                    if valid_mask.any():
                        sample_loss = loss_per_token[i][valid_mask].mean().item()
                    else:
                        sample_loss = 0.0
                    losses.append(sample_loss)

        model.train()

        # Clean up memory after loss computation
        del dataloader, dataset_for_loss
        torch.cuda.empty_cache()

        # Cache the losses
        if cache_path:
            print(f"Saving computed losses to {cache_path}...")
            try:
                with open(cache_path, 'w') as f:
                    json.dump(losses, f)
                print("Losses saved successfully")
            except Exception as e:
                print(f"Error saving losses: {e}")

    # Create list of (index, loss) and sort by loss
    indexed_losses = list(enumerate(losses))
    indexed_losses.sort(key=lambda x: x[1], reverse=reverse)

    # Select top num_samples
    selected_indices = [idx for idx, _ in indexed_losses[:num_samples]]

    order_str = "highest" if reverse else "lowest"
    print(f"Selected {len(selected_indices)} samples with {order_str} losses")
    print(f"  Loss range: {indexed_losses[0][1]:.4f} to {indexed_losses[min(num_samples-1, len(indexed_losses)-1)][1]:.4f}")

    return dataset.select(selected_indices)

def select_phase_a_data(dataset, num_samples, method="random", **kwargs):
    """
    Select samples from Phase A dataset using specified method.

    Args:
        dataset: The tokenized dataset to select from
        num_samples: Number of samples to select
        method: Selection method ("random", "difficulty_hard", "difficulty_easy", "loss_descending", or "loss_ascending")
        **kwargs: Additional arguments to pass to selection functions
                  (e.g., model, tokenizer, raw_dataset)

    Returns:
        Selected subset of the dataset
    """
    print(f"\n--- Selecting {num_samples} samples from Phase A using '{method}' method ---")

    if method == "random":
        return select_random(dataset, num_samples, **kwargs)
    elif method == "difficulty_hard":
        return select_by_difficulty(dataset, num_samples, reverse=True, **kwargs)
    elif method == "difficulty_easy":
        return select_by_difficulty(dataset, num_samples, reverse=False, **kwargs)
    elif method == "loss_descending":
        return select_by_loss(dataset, num_samples, reverse=True, **kwargs)
    elif method == "loss_ascending":
        return select_by_loss(dataset, num_samples, reverse=False, **kwargs)
    else:
        print(f"WARNING: Unknown selection method '{method}'. Falling back to random selection.")
        return select_random(dataset, num_samples, **kwargs)

# --- 4. Main Experiment Logic ---
def main(selection_method="difficulty"):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print(f"--- Loading Base Model & Tokenizer ---")
    print(f"Selection Method: {selection_method}")
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


    # --- 【已反转】Experiment 2: Sequential Training (CF) [MATH -> HotpotQA] ---
    print(f"\n--- Starting Experiment 2: Sequential Training (CF) [MATH -> HotpotQA] ---")

    # --- Phase 1: Train on MATH (or load from checkpoint) ---
    if os.path.exists(os.path.join(TASK_A_ADAPTER_PATH, "adapter_model.safetensors")):
        print(f"--- Found existing Task A (MATH) adapter. Loading from {TASK_A_ADAPTER_PATH} ---")
        seq_model = PeftModel.from_pretrained(base_model, TASK_A_ADAPTER_PATH, is_trainable=True)
        print("Adapter loaded successfully.")

    else:
        print(f"--- No adapter found. Starting Phase 1: Training on Task A (MATH) ---")
        lora_config = get_lora_config()
        seq_model = get_peft_model(base_model, lora_config)
        seq_model.print_trainable_parameters()

        seq_args_a = TrainingArguments(
            output_dir=os.path.join(RESULTS_DIR, "seq_training_A"),
            per_device_train_batch_size=PER_DEVICE_BS,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            num_train_epochs=TASK_A_EPOCHS,
            learning_rate=2e-4,
            bf16=True, # <-- 【BUG 修复】删除此行
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            gradient_checkpointing=True,
        )

        seq_trainer_a = Trainer(
            model=seq_model,
            args=seq_args_a,
            train_dataset=math_train_tokenized, # <-- 训练 MATH
            eval_dataset=math_val_tokenized,
            data_collator=data_collator,
        )

        seq_trainer_a.train()

        print(f"--- Phase 1 (MATH) training complete. Saving adapter to {TASK_A_ADAPTER_PATH} ---")
        seq_model.save_pretrained(TASK_A_ADAPTER_PATH)
        print("Adapter saved.")

        del seq_trainer_a
        torch.cuda.empty_cache()

     # --- Evaluate the "Task A Expert" model (whether trained or loaded) ---
    print("\n--- Evaluating Model after Phase 1 (Task A Expert) ---")
    eval_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "eval_temp"),
        per_device_eval_batch_size=PER_DEVICE_BS,
        bf16=True,
        report_to="none",
        gradient_checkpointing=True,
    )


    eval_trainer = Trainer(
        model=seq_model,
        args=eval_args,
        data_collator=data_collator,
    )

    eval_hotpot_phase1 = eval_trainer.evaluate(eval_dataset=hotpot_val_tokenized)
    print(f"  > Task B Expert - HotpotQA Val Loss: {eval_hotpot_phase1['eval_loss']:.4f}")
    eval_math_phase1 = eval_trainer.evaluate(eval_dataset=math_val_tokenized)
    print(f"  > Task A Expert - MATH Val Loss: {eval_math_phase1['eval_loss']:.4f}")
    del eval_trainer, eval_args
    torch.cuda.empty_cache()

    # --- Phase 2: Train on HotpotQA with Data Mixing ---
    print(f"\n  --- Phase 2: Training on Task B (HotpotQA) with Data Mixing ---")

    # Calculate number of Phase A samples to mix in
    num_phase_a_samples = int(len(hotpot_train_tokenized) * MIX_PERCENTAGE)
    print(f"Mixing in {num_phase_a_samples} samples ({MIX_PERCENTAGE*100:.0f}%) from Phase A (MATH)")

    # Select Phase A data based on configured method
    if MIX_PERCENTAGE > 0:
        selected_math_data = select_phase_a_data(
            dataset=math_train_tokenized,
            num_samples=num_phase_a_samples,
            method=selection_method,
            model=seq_model,
            tokenizer=tokenizer,
            raw_dataset=math_train,  # For difficulty-based selection
            cache_dir=RESULTS_DIR  # For loss-based selection caching
        )

        # Combine Phase B data with selected Phase A data
        mixed_train_dataset = concatenate_datasets([hotpot_train_tokenized, selected_math_data])
        print(f"Mixed dataset size: {len(mixed_train_dataset)} (HotpotQA: {len(hotpot_train_tokenized)}, MATH: {len(selected_math_data)})")

        # Shuffle the mixed dataset
        mixed_train_dataset = mixed_train_dataset.shuffle(seed=42)
    else:
        print("MIX_PERCENTAGE is 0, using only HotpotQA data (no mixing)")
        mixed_train_dataset = hotpot_train_tokenized

    history = {
        "steps": [],
        "hotpot_loss": [],
        "math_loss": [],
        "selection_method": selection_method,
        "mix_percentage": MIX_PERCENTAGE
    }
    # Custom Trainer to log forgetting
    class ForgettingTrackerCallback(TrainerCallback):
      def __init__(self, hotpot_val, math_val, history_log, start_metrics):
          super().__init__()
          self.hotpot_eval_dataset = hotpot_val
          self.math_eval_dataset = math_val
          self.history = history_log
          self.trainer = None
          # --- 【修复】---
          # 添加一个 "锁" 来防止无限递归
          self.is_evaluating = False
          # ----------------
          # 记录初始状态 (Step 0)
          self.history["steps"].append(0)
          self.history["hotpot_loss"].append(start_metrics['hotpot_loss'])
          self.history["math_loss"].append(start_metrics['math_loss'])
          print("Initializing ForgettingTrackerCallback with starting metrics.")
      def set_trainer(self, trainer):
          """在 Trainer 例化后, 注入对它的引用。"""
          self.trainer = trainer
          print("Trainer reference set in callback.")

      def on_log(self, args, state, control, **kwargs):
          """在 'logging_steps' 触发时被调用。"""
          # --- 【修复 1】---
          # 如果我们已经在这个函数中 (因为递归调用), 立即退出。
          if self.is_evaluating:
              return
          # --- 【修复 2】---
          # "获取" 锁
          self.is_evaluating = True
          # 确保 trainer 引用已被设置
          if not self.trainer:
              print("WARNING: Trainer reference not set in callback, skipping eval.")
              self.is_evaluating = False # <-- 别忘了在这里释放锁
              return
          print(f"\n--- Custom Eval at Step {state.global_step} ---")
          print("Evaluating on Task B (HotpotQA)...")
          # 使用 trainer 的 evaluate 方法
          hotpot_metrics = self.trainer.evaluate(eval_dataset=self.hotpot_eval_dataset)
          hotpot_loss = hotpot_metrics['eval_loss']
          print(f"  > Step {state.global_step} - HotpotQA Val Loss: {hotpot_loss:.4f} (LEARNING?)")
          print("Evaluating on Task A (MATH)...")
          math_metrics = self.trainer.evaluate(eval_dataset=self.math_eval_dataset)
          math_loss = math_metrics['eval_loss']
          print(f"  > Step {state.global_step} - MATH Val Loss: {math_loss:.4f} (FORGETTING?)")
          self.history["steps"].append(state.global_step)
          self.history["hotpot_loss"].append(hotpot_loss)
          self.history["math_loss"].append(math_loss)
          # --- 【修复 3】---
          # "释放" 锁, 以便下一次 on_log 可以运行
          self.is_evaluating = False
          self.trainer.model.train()


    seq_args_b = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "seq_training_B"),
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_train_epochs=TASK_B_EPOCHS,
        learning_rate=7e-5,
        logging_steps=10,
        save_strategy="no",
        report_to=[],         # <-- 保持这个设置
        # disable_tqdm=True,  # <-- 保持这个设置
        gradient_checkpointing=True,
        bf16=True
    )
    seq_model.enable_input_require_grads()
    # 【修复 2】: 实例化 *新* 的 Callback
    tracker_callback = ForgettingTrackerCallback(
        hotpot_val=hotpot_val_tokenized,
        math_val=math_val_tokenized,
        history_log=history,
        start_metrics={
            'hotpot_loss': eval_hotpot_phase1['eval_loss'],
            'math_loss': eval_math_phase1['eval_loss'],
        }
    )
    # 【修复 3】: 实例化一个 *标准* Trainer, 并传入回调
    seq_trainer_b = Trainer(
        model=seq_model,
        args=seq_args_b,
        train_dataset=mixed_train_dataset,  # <-- Use mixed dataset instead of hotpot_train_tokenized
        eval_dataset=hotpot_val_tokenized,
        data_collator=data_collator,
        callbacks=[tracker_callback]  # <-- 在这里传入回调
    )
    # 【修复 4】: 将 trainer 实例链接回回调
    # (回调需要这个引用来调用 self.trainer.evaluate())
    tracker_callback.set_trainer(seq_trainer_b)
    seq_trainer_b.train()

    # --- 5. Plot Results ---
    print("\n--- Saving History Data and Generating Plot ---")

    # --- 保存 history data 到 JSON ---
    # Include mixing method and percentage in the filename
    mix_pct_str = f"{int(MIX_PERCENTAGE * 100)}pct"
    history_filename = get_unique_filename(
        RESULTS_DIR,
        f"forgetting_history_MATH_to_HotpotQA_{selection_method}_{mix_pct_str}_bf16",
        ".json"
    )
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

    plt.title(f"Catastrophic Forgetting: MATH -> HotpotQA (Model: {MODEL_NAME} BF16 LoRA)\nSelection: {selection_method} | Mix: {MIX_PERCENTAGE*100:.0f}%")
    plt.xlabel(f"Training Steps on Task B (HotpotQA) (Total Epochs: {TASK_B_EPOCHS})")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()

    plot_filename = get_unique_filename(
        RESULTS_DIR,
        f"sequential_forgetting_curve_MATH_to_HotpotQA_{selection_method}_{mix_pct_str}_bf16",
        ".png"
    )
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    try:
        from google.colab import files
        plt.show()
    except ImportError:
        print("Not in Colab, plot saved to file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with catastrophic forgetting mitigation')
    parser.add_argument(
        '--selection-method',
        type=str,
        default='difficulty_hard',
        choices=['random', 'difficulty_hard', 'difficulty_easy', 'loss_descending', 'loss_ascending'],
        help='Method to select Phase A data for mixing (default: difficulty_hard)'
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: This experiment requires a GPU. Check Colab runtime type.")
    else:
        print(f"INFO: Running on GPU. VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        if torch.cuda.get_device_properties(0).total_memory / 1e9 < 11:
            print("WARNING: VRAM is less than 11GB. You may hit OOM errors. Try lowering MAX_SEQ_LENGTH.")

    main(selection_method=args.selection_method)