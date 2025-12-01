# !pip install tf-keras
import sys
import torch
import json
import os
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments
)
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import warnings
from torch.utils.data import DataLoader

from torch.optim import AdamW

is_colab = False
try:
    from google.colab import drive
    drive.mount("/content/drive")
    is_colab = True
except ImportError:
    print("Not in Colab, skipped mount drive.")

warnings.filterwarnings("ignore")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HOTPOT_DATASET_NAME = "hotpot_qa"
HOTPOT_DATASET_CONFIG = "distractor"
MATH_DATASET_NAME = "qwedsacf/competition_math"
RESULTS_DIR = "results" if not is_colab else "/content/drive/MyDrive/cs182_experiments/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
TASK_A_ADAPTER_PATH = os.path.join(RESULTS_DIR, "math_adapter_llama_fp32")
TASK_A_EPOCHS = 8
TASK_B_EPOCHS = 8

MAX_SEQ_LENGTH = 512
PER_DEVICE_BS = 4 # This will be the batch size for *each* task's DataLoader
GRAD_ACC_STEPS = 1 # Kept for consistency, but manual loop handles 1 step at a time

N_TRAIN_EXAMPLES = 4000
N_VAL_EXAMPLES = 400
LEARNING_RATE = 7e-5       # Using the rate from the original Task A / Joint training
INTERLEAVE_EPOCHS = 4      # Doubled epochs (was 2 in baseline)
LOGGING_STEPS = 10         # Evaluate every 10 *batches* (e.g., 5 MATH, 5 HotpotQA)
SMOOTHING_COEFFS = [0, 0.05, 0.1, 0.15, 0.2]
prev_loss = None

import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename=os.path.join(RESULTS_DIR, "logger.txt"),          # log file name
    filemode="a",                     # append log history
    level=logging.INFO,               # or DEBUG for even more detail
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

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
    """
    只加载 FP16 TinyLlama 基础模型和 Tokenizer。
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16, # <-- 加载时仍然用 FP16 (节省 RAM)，但训练会是 FP32
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

class SmoothedLossTrainer(Trainer):
    def __init__(self, *args, alpha=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.prev_loss = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = None, None
        # standard HF loss only, no logits returned
        return_value = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch
        )
        if return_outputs:
          loss, outputs = return_value
        else:
          loss = return_value
        if self.prev_loss is None:
            self.prev_loss = loss.detach()
        smooth_loss = (1 - self.alpha) * loss + self.alpha * self.prev_loss
        self.prev_loss = loss.detach()
        return (smooth_loss, outputs) if return_outputs else smooth_loss
    
def main(alpha:float):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    logger.info(f"--- Loading Base Model & Tokenizer ---")
    print(f"--- Loading Base Model & Tokenizer ---")

    base_model, tokenizer = get_model_and_tokenizer_base()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Load and process data
    logger.info(f"\n--- Loading and Preprocessing Datasets (This may take a while) ---")
    print(f"\n--- Loading and Preprocessing Datasets (This may take a while) ---")

    # Task B - HOTPOT QA
    raw_hotpot = load_dataset(HOTPOT_DATASET_NAME, HOTPOT_DATASET_CONFIG)
    hotpot_train = raw_hotpot["train"].shuffle(seed=42).select(range(N_TRAIN_EXAMPLES))
    hotpot_val = raw_hotpot["validation"].shuffle(seed=42).select(range(N_VAL_EXAMPLES))

    logger.info(f"Tokenize and filter the HotPotQA dataset... ")
    print("Tokenize and filter the HotPotQA dataset... ")
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

    logger.info(f"HotpotQA: {len(hotpot_train_tokenized)} train, {len(hotpot_val_tokenized)} val (after filtering)")
    print(f"HotpotQA: {len(hotpot_train_tokenized)} train, {len(hotpot_val_tokenized)} val (after filtering)")

    # Task A: MATH
    raw_math = load_dataset(MATH_DATASET_NAME)
    total_math_samples_needed = N_TRAIN_EXAMPLES + N_VAL_EXAMPLES
    math_subset = raw_math["train"].shuffle(seed=42).select(range(total_math_samples_needed))
    val_size_fraction = N_VAL_EXAMPLES / total_math_samples_needed
    math_splits = math_subset.train_test_split(test_size=val_size_fraction, seed=42)
    math_train = math_splits["train"]
    math_val = math_splits["test"]

    logger.info(f"Tokenizing and Filtering Math... ")
    print(f"Tokenizing and Filtering Math... ")

    math_train_tokenized = math_train.filter(
        lambda x: filter_by_length(x, tokenizer, format_math),
        batched=False,
    ).map(
        lambda x: preprocess(x, tokenizer, format_math),
        batched=False,
    ).filter(lambda example:len(example) > 0)
    math_val_tokenized = math_val.filter(
        lambda x: filter_by_length(x, tokenizer, format_math),
        batched=False,
    ).map(
        lambda x: preprocess(x, tokenizer, format_math),
        batched=False,
    ).filter(lambda example: len(example) > 0)

    logger.info(f"MATH: {len(math_train_tokenized)} train, {len(math_val_tokenized)} val (after filtering)")
    print(f"MATH: {len(math_train_tokenized)} train, {len(math_val_tokenized)} val (after filtering)")

    logger.info(f"  Phase1 training: {len(math_train_tokenized)} samples")
    logger.info(f"  Phase2 training: {len(hotpot_train_tokenized)} samples")
    print(f"  Phase1 training: {len(math_train_tokenized)} samples")
    print(f"  Phase2 training: {len(hotpot_train_tokenized)} samples")

    # --- 【已反转】Experiment 2: Sequential Training (CF) [MATH -> HotpotQA] ---
    logger.info("\n--- Starting Experiment 2: Sequential Training (CF) [MATH -> HotpotQA] ---")
    print(f"\n--- Starting Experiment 2: Sequential Training (CF) [MATH -> HotpotQA] ---")

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
          logger.info("Initializing ForgettingTrackerCallback with starting metrics.")
          print("Initializing ForgettingTrackerCallback with starting metrics.")

      def set_trainer(self, trainer):
          """在 Trainer 例化后, 注入对它的引用。"""
          self.trainer = trainer
          logger.info("Trainer reference set in callback.")
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
              logger.info("WARNING: Trainer reference not set in callback, skipping eval.")
              print("WARNING: Trainer reference not set in callback, skipping eval.")
              self.is_evaluating = False # <-- 别忘了在这里释放锁
              return

          logger.info(f"\n--- Custom Eval at Step {state.global_step} ---")
          logger.info("Evaluating on Task B (HotpotQA)...")
          print(f"\n--- Custom Eval at Step {state.global_step} ---")
          print("Evaluating on Task B (HotpotQA)...")

          # 使用 trainer 的 evaluate 方法
          hotpot_metrics = self.trainer.evaluate(eval_dataset=self.hotpot_eval_dataset)
          hotpot_loss = hotpot_metrics['eval_loss']

          logger.info(f"  > Step {state.global_step} - HotpotQA Val Loss: {hotpot_loss:.4f} (LEARNING?)")
          logger.info("Evaluating on Task A (MATH)...")
          print(f"  > Step {state.global_step} - HotpotQA Val Loss: {hotpot_loss:.4f} (LEARNING?)")
          print("Evaluating on Task A (MATH)...")

          math_metrics = self.trainer.evaluate(eval_dataset=self.math_eval_dataset)
          math_loss = math_metrics['eval_loss']

          logger.info(f"  > Step {state.global_step} - MATH Val Loss: {math_loss:.4f} (FORGETTING?)")
          print(f"  > Step {state.global_step} - MATH Val Loss: {math_loss:.4f} (FORGETTING?)")

          self.history["steps"].append(state.global_step)
          self.history["hotpot_loss"].append(hotpot_loss)
          self.history["math_loss"].append(math_loss)
          # --- 【修复 3】---
          # "释放" 锁, 以便下一次 on_log 可以运行
          self.is_evaluating = False
          self.trainer.model.train()


    # --- Phase 1: Train on MATH (or load from checkpoint) ---
    logger.info(f"--- Going to begin Phase 1: Training on Task A (MATH) ---")
    print(f"--- Going to begin Phase 1: Training on Task A (MATH) ---")
    lora_config = get_lora_config()
    seq_model = get_peft_model(base_model, lora_config)
    seq_model.print_trainable_parameters()

    # disable checkpointing
    # seq_model.gradient_checkpointing_disable()
    # seq_model.config.use_cache = True

    seq_args_a = TrainingArguments(
            output_dir=os.path.join(RESULTS_DIR, "seq_training_A"),
            per_device_train_batch_size=PER_DEVICE_BS,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            num_train_epochs=TASK_A_EPOCHS,
            learning_rate=2e-4,
            # fp16=True, # <-- 【BUG 修复】删除此行
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            gradient_checkpointing=True,
    )

    seq_trainer_a = SmoothedLossTrainer(
            model=seq_model,
            args=seq_args_a,
            train_dataset=math_train_tokenized, # <-- 训练 MATH
            eval_dataset=math_val_tokenized,
            data_collator=data_collator,
            alpha=alpha,
    )

    seq_trainer_a.train()

    logger.info(f"--- Phase 1 (MATH) training complete. Saving adapter to {TASK_A_ADAPTER_PATH} ---")
    print(f"--- Phase 1 (MATH) training complete. Saving adapter to {TASK_A_ADAPTER_PATH} ---")
    # seq_trainer_a.save_model()
    seq_model.save_pretrained(TASK_A_ADAPTER_PATH)
    logger.info("Adapter saved.")
    print("Adapter saved.")

    #保留内存 （save memory)
    del seq_trainer_a
    torch.cuda.empty_cache()

     # --- Evaluate the "Task A Expert" model (whether trained or loaded) ---
    logger.info("\n--- Evaluating Model after Phase 1 (Task A Expert) ---")
    print("\n--- Evaluating Model after Phase 1 (Task A Expert) ---")
    eval_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "eval_temp"),
        per_device_eval_batch_size=PER_DEVICE_BS,
        # fp16=True, # <-- 【BUG 修复】删除此行
        report_to="none",
        gradient_checkpointing=True,
    )


    eval_trainer = Trainer(
        model=seq_model,
        args=eval_args,
        data_collator=data_collator,
    )

    eval_hotpot_phase1 = eval_trainer.evaluate(eval_dataset=hotpot_val_tokenized)
    logger.info(f"  > Task B Expert - HotpotQA Val Loss: {eval_hotpot_phase1['eval_loss']:.4f}")
    print(f"  > Task B Expert - HotpotQA Val Loss: {eval_hotpot_phase1['eval_loss']:.4f}")

    eval_math_phase1 = eval_trainer.evaluate(eval_dataset=math_val_tokenized)
    logger.info(f"  > Task A Expert - MATH Val Loss: {eval_math_phase1['eval_loss']:.4f}")
    print(f"  > Task A Expert - MATH Val Loss: {eval_math_phase1['eval_loss']:.4f}")

    #保留内存 （save memory)
    del eval_trainer, eval_args
    torch.cuda.empty_cache()

    # --- Phase 2: Train on HotpotQA (Forgetting MATH happens here) ---
    logger.info(f"\n  --- Phase 2: Training on Task B (HotpotQA) ---")
    print(f"\n  --- Phase 2: Training on Task B (HotpotQA) ---")
    history = {"steps": [], "hotpot_loss": [], "math_loss": []}

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
    seq_trainer_b = SmoothedLossTrainer(
        model=seq_model,
        args=seq_args_b,
        train_dataset=hotpot_train_tokenized,
        eval_dataset=hotpot_val_tokenized,
        data_collator=data_collator,
        callbacks=[tracker_callback], # <-- 在这里传入回调
        alpha=alpha,
    )
    # 【修复 4】: 将 trainer 实例链接回回调
    # (回调需要这个引用来调用 self.trainer.evaluate())
    tracker_callback.set_trainer(seq_trainer_b)
    seq_trainer_b.train()

    # --- 5. Plot Results ---
    logger.info("\n--- Saving History Data and Generating Plot ---")
    print("\n--- Saving History Data and Generating Plot ---")

    # --- 保存 history data 到 JSON ---
    history_filename = os.path.join(RESULTS_DIR, f"forgetting_curve_MATH_alpha_{int(alpha*100)}_percent.json")
    try:
        with open(history_filename, 'w') as f:
            json.dump(history, f, indent=4)
        logger.info(f"History data saved to {history_filename}")
        print(f"History data saved to {history_filename}")
    except Exception as e:
        logger.info(f"Error saving history to JSON: {e}")
        print(f"Error saving history to JSON: {e}")
    # --- [END] ---

    plt.figure(figsize=(12, 6))
    plt.plot(history["steps"], history["hotpot_loss"], 'o-', label="Task B (HotpotQA) Loss", color="blue")
    plt.plot(history["steps"], history["math_loss"], 'o-', label="Task A (MATH) Loss", color="red")

    plt.title(f"Catastropic Forgetting: Math -> HotPotQA (Model: {MODEL_NAME} FP32 LoRA)")
    plt.xlabel(f"Training Steps on Task B (HotpotQA) (Total Epochs: {TASK_B_EPOCHS})")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()

    plot_filename = os.path.join(RESULTS_DIR, f"sequential_forgetting_curve_MATH_to_HotpotQA_fp32_alpha_{alpha}.png")
    plt.savefig(plot_filename)
    logger.info(f"Plot saved to {plot_filename}")
    print(f"Plot saved to {plot_filename}")

    #保留内存 （save memory)
    del seq_trainer_b
    torch.cuda.empty_cache()


    try:
        from google.colab import files
        plt.show()
    except ImportError:
        logger.info("Not in Colab, plot saved to file.")
        print("Not in Colab, plot saved to file.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        logger.info("ERROR: This experiment requires a GPU. Check Colab runtime type.")
        print("ERROR: This experiment requires a GPU. Check Colab runtime type.")
    else:
        logger.info(f"INFO: Running on GPU. VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"INFO: Running on GPU. VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        if torch.cuda.get_device_properties(0).total_memory / 1e9 < 11:
            logger.info("WARNING: VRAM is less than 11GB. You may hit OOM errors. Try lowering MAX_SEQ_LENGTH.")
            print("WARNING: VRAM is less than 11GB. You may hit OOM errors. Try lowering MAX_SEQ_LENGTH.")
    for handler in logger.handlers:
        handler.flush()
    main(0)
