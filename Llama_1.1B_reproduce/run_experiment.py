# run_experiment.py
import torch
import os
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BitsAndBytesConfig, # <-- 【已移除】
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk, concatenate_datasets
# 【已移除】 prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model 
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. Configuration ---
ASSETS_DIR = "./offline_assets"
# 【模型路径】指向 Llama 1.1B
MODEL_PATH = os.path.join(ASSETS_DIR, "TinyLlama_TinyLlama-1.1B-Chat-v1.0")
HOTPOT_PATH = os.path.join(ASSETS_DIR, "hotpot_qa")
MATH_PATH = os.path.join(ASSETS_DIR, "qwedsacf_competition_math") 
RESULTS_DIR = "./results"

# 【变更】新的适配器路径
TASK_A_ADAPTER_PATH = os.path.join(RESULTS_DIR, "hotpotqa_adapter_llama_fp16") 

# --- VRAM-Saving Config ---
MAX_SEQ_LENGTH = 1024 # 如果 OOM, 降低到 768
PER_DEVICE_BS = 1  # 必须为 1
GRAD_ACC_STEPS = 16 # 模拟 BS=16

# --- Experiment Config ---
N_TRAIN_EXAMPLES = 4000 
N_VAL_EXAMPLES = 400
JOINT_EPOCHS = 2
TASK_A_EPOCHS = 2
TASK_B_EPOCHS = 2 

# --- 2. Utility Functions (Data Formatting - Llama Chat Style) ---
# (此部分没有变化, TinyLlama 是聊天模型)
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

def preprocess_and_filter(example, tokenizer, formatter):
    """
    Formats, tokenizes, and filters the example based on length.
    """
    text = formatter(example) 
    
    tokenized = tokenizer(text, max_length=MAX_SEQ_LENGTH + 1, truncation=False, padding=False)

    if len(tokenized['input_ids']) > MAX_SEQ_LENGTH:
        return {} 

    tokenized = tokenizer(
        text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length", # Pad to max_length
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# --- 3. Model Loading (【变更】标准 LoRA, 无 4-bit) ---
def get_model_and_tokenizer():
    """Loads the FP16 TinyLlama model with standard LoRA adapters."""
    
    # 【已移除】 bnb_config
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16, # <-- 【变更】以 FP16 加载
        device_map="auto", # 自动映射到 GPU
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 
    
    # --- PEFT & LoRA Config ---
    model.gradient_checkpointing_enable() # <-- 关键的显存优化
    # 【已移除】 prepare_model_for_kbit_training
    
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16,
        target_modules=[ # Llama 模块
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters() 
    return model, tokenizer

# --- 4. Main Experiment Logic ---
# (此部分没有变化)
def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    print(f"--- Loading Base Model & Tokenizer ---")
    base_model, tokenizer = get_model_and_tokenizer()
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Load and Process Datasets ---
    print(f"\n--- Loading and Preprocessing Datasets (This may take a while) ---")
    
    # Task A: HotpotQA
    raw_hotpot = load_from_disk(HOTPOT_PATH)
    hotpot_train = raw_hotpot["train"].shuffle(seed=42).select(range(N_TRAIN_EXAMPLES))
    hotpot_val = raw_hotpot["validation"].shuffle(seed=42).select(range(N_VAL_EXAMPLES))
    
    print(f"Tokenizing and filtering HotpotQA...")
    hotpot_train_tokenized = hotpot_train.map(
        lambda x: preprocess_and_filter(x, tokenizer, format_hotpot_qa),
        batched=False, 
    ).filter(lambda example: len(example.get('input_ids', [])) > 0)
    
    hotpot_val_tokenized = hotpot_val.map(
        lambda x: preprocess_and_filter(x, tokenizer, format_hotpot_qa),
        batched=False,
    ).filter(lambda example: len(example.get('input_ids', [])) > 0)
    
    print(f"HotpotQA: {len(hotpot_train_tokenized)} train, {len(hotpot_val_tokenized)} val (after filtering)")

    # Task B: MATH
    raw_math = load_from_disk(MATH_PATH) 
    math_train = raw_math["train"].shuffle(seed=42).select(range(N_TRAIN_EXAMPLES))
    math_val = raw_math["test"].shuffle(seed=42).select(range(N_VAL_EXAMPLES))
    
    print(f"Tokenizing and filtering MATH...")
    math_train_tokenized = math_train.map(
        lambda x: preprocess_and_filter(x, tokenizer, format_math),
        batched=False,
    ).filter(lambda example: len(example.get('input_ids', [])) > 0)
    
    math_val_tokenized = math_val.map(
        lambda x: preprocess_and_filter(x, tokenizer, format_math),
        batched=False,
    ).filter(lambda example: len(example.get('input_ids', [])) > 0)

    print(f"MATH: {len(math_train_tokenized)} train, {len(math_val_tokenized)} val (after filtering)")

    # --- Experiment 1: Joint Training (Control Group) ---
    print(f"\n--- Starting Experiment 1: Joint Training ---")
    
    joint_train_dataset = concatenate_datasets([hotpot_train_tokenized, math_train_tokenized]).shuffle(seed=42)
    joint_model, _ = get_model_and_tokenizer()
    
    joint_training_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "joint_training"),
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_train_epochs=JOINT_EPOCHS,
        learning_rate=2e-4,
        fp16=True, # 必须为 True
        logging_steps=50,
        save_strategy="no", 
        report_to="none", 
    )
    
    joint_trainer = Trainer(
        model=joint_model,
        args=joint_training_args,
        train_dataset=joint_train_dataset,
        data_collator=data_collator,
    )
    
    joint_trainer.train()
    
    print("\n--- Evaluating Joint Model ---")
    eval_hotpot_joint = joint_trainer.evaluate(eval_dataset=hotpot_val_tokenized)
    print(f"  > Joint Model - HotpotQA Val Loss: {eval_hotpot_joint['eval_loss']:.4f}")
    
    eval_math_joint = joint_trainer.evaluate(eval_dataset=math_val_tokenized)
    print(f"  > Joint Model - MATH Val Loss: {eval_math_joint['eval_loss']:.4f}")
    
    del joint_model, joint_trainer
    torch.cuda.empty_cache()


    # --- Experiment 2: Sequential Training (Forgetting) ---
    print(f"\n--- Starting Experiment 2: Sequential Training (CF) ---")
    
    seq_model, _ = get_model_and_tokenizer()
    
    # --- Phase 1: Train on HotpotQA (or load from checkpoint) ---
    if os.path.exists(os.path.join(TASK_A_ADAPTER_PATH, "adapter_model.bin")):
        print(f"--- Found existing Task A adapter. Loading from {TASK_A_ADAPTER_PATH} ---")
        seq_model.load_adapter(TASK_A_ADAPTER_PATH)
        print("Adapter loaded successfully.")
    
    else:
        print(f"--- No adapter found. Starting Phase 1: Training on Task A (HotpotQA) ---")
        
        seq_args_a = TrainingArguments(
            output_dir=os.path.join(RESULTS_DIR, "seq_training_A_temp"), 
            per_device_train_batch_size=PER_DEVICE_BS,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            num_train_epochs=TASK_A_EPOCHS,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=50,
            save_strategy="no", 
            report_to="none",
        )
        
        seq_trainer_a = Trainer(
            model=seq_model,
            args=seq_args_a,
            train_dataset=hotpot_train_tokenized,
            eval_dataset=hotpot_val_tokenized,
            data_collator=data_collator,
        )
        
        seq_trainer_a.train()
        
        print(f"--- Phase 1 training complete. Saving adapter to {TASK_A_ADAPTER_PATH} ---")
        seq_model.save_adapter(TASK_A_ADAPTER_PATH)
        print("Adapter saved.")
        
        del seq_trainer_a
        torch.cuda.empty_cache()

    # --- Evaluate the "Task A Expert" model (whether trained or loaded) ---
    print("\n--- Evaluating Model after Phase 1 (Task A Expert) ---")
    
    eval_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "eval_temp"),
        per_device_eval_batch_size=PER_DEVICE_BS * 2, 
        fp16=True,
        report_to="none",
    )
    
    eval_trainer = Trainer(
        model=seq_model,
        args=eval_args,
        data_collator=data_collator,
    )
    
    eval_hotpot_phase1 = eval_trainer.evaluate(eval_dataset=hotpot_val_tokenized)
    print(f"  > Task A Expert - HotpotQA Val Loss: {eval_hotpot_phase1['eval_loss']:.4f}")
    eval_math_phase1 = eval_trainer.evaluate(eval_dataset=math_val_tokenized)
    print(f"  > Task A Expert - MATH Val Loss: {eval_math_phase1['eval_loss']:.4f}")
    
    del eval_trainer, eval_args
    torch.cuda.empty_cache()


    # --- Phase 2: Train on MATH (Forgetting happens here) ---
    print(f"\n  --- Phase 2: Training on Task B (MATH) ---")
    
    history = {"steps": [], "hotpot_loss": [], "math_loss": []}
    
    # Custom Trainer to log forgetting
    class ForgettingTrackerTrainer(Trainer):
        def __init__(self, start_metrics, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.hotpot_eval_dataset = hotpot_val_tokenized
            self.math_eval_dataset = math_val_tokenized
            self.history = history
            self.history["steps"].append(0)
            self.history["hotpot_loss"].append(start_metrics['hotpot_loss'])
            self.history["math_loss"].append(start_metrics['math_loss'])

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % 100 == 0: 
                print(f"\n--- Custom Eval at Step {state.global_step} ---")
                self.model.eval() 
                
                print("Evaluating on Task A (HotpotQA)...")
                hotpot_metrics = self.evaluate(eval_dataset=self.hotpot_eval_dataset)
                hotpot_loss = hotpot_metrics['eval_loss']
                print(f"  > Step {state.global_step} - HotpotQA Val Loss: {hotpot_loss:.4f} (FORGETTING?)")
                
                print("Evaluating on Task B (MATH)...")
                math_metrics = self.evaluate(eval_dataset=self.math_eval_dataset)
                math_loss = math_metrics['eval_loss']
                print(f"  > Step {state.global_step} - MATH Val Loss: {math_loss:.4f} (LEARNING?)")
                
                self.history["steps"].append(state.global_step)
                self.history["hotpot_loss"].append(hotpot_loss)
                self.history["math_loss"].append(math_loss)

    seq_args_b = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "seq_training_B"),
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_train_epochs=TASK_B_EPOCHS,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
    )

    seq_trainer_b = ForgettingTrackerTrainer(
        start_metrics={ 
            'hotpot_loss': eval_hotpot_phase1['eval_loss'],
            'math_loss': eval_math_phase1['eval_loss'],
        },
        model=seq_model, 
        args=seq_args_b,
        train_dataset=math_train_tokenized,
        eval_dataset=math_val_tokenized, 
        data_collator=data_collator,
    )
    
    seq_trainer_b.train()
    
    # --- 5. Plot Results ---
    print("\n--- Generating Loss Curve Plot ---")
    
    plt.figure(figsize=(12, 6))
    plt.plot(history["steps"], history["hotpot_loss"], 'o-', label="Task A (HotpotQA) Loss", color="red")
    plt.plot(history["steps"], history["math_loss"], 'o-', label="Task B (MATH) Loss", color="blue")
    
    plt.title(f"Catastrophic Forgetting: HotpotQA -> MATH (Model: {MODEL_PATH} FP16 LoRA)")
    plt.xlabel(f"Training Steps on Task B (MATH) (Total Epochs: {TASK_B_EPOCHS})")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.yscale('log') 
    plt.tight_layout()
    
    plot_filename = os.path.join(RESULTS_DIR, "sequential_forgetting_curve_fp16.png")
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: This experiment requires a GPU.")
    else:
        print(f"INFO: Running on GPU. VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        if torch.cuda.get_device_properties(0).total_memory / 1e9 < 11:
            print("WARNING: VRAM is less than 11GB. You may hit OOM errors. Try lowering MAX_SEQ_LENGTH.")
    main()