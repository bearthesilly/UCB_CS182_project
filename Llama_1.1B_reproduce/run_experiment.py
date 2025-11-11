import torch
import os
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
# 【已移除】 prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model 
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HOTPOT_DATASET_NAME = "hotpot_qa"
HOTPOT_DATASET_CONFIG = "distractor"
MATH_DATASET_NAME = "qwedsacf/competition_math"
RESULTS_DIR = "./results"

# 【检查点路径 1】
TASK_A_ADAPTER_PATH = os.path.join(RESULTS_DIR, "hotpotqa_adapter_llama_fp16") 

# 【新变更】为 Joint Training 添加检查点路径
JOINT_ADAPTER_PATH = os.path.join(RESULTS_DIR, "joint_adapter_llama_fp16")

# --- VRAM-Saving Config ---
MAX_SEQ_LENGTH = 1024 
PER_DEVICE_BS = 1  
GRAD_ACC_STEPS = 16 

# --- Experiment Config ---
N_TRAIN_EXAMPLES = 4000 
N_VAL_EXAMPLES = 400
JOINT_EPOCHS = 2
TASK_A_EPOCHS = 2
TASK_B_EPOCHS = 2 

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

def preprocess(example, tokenizer, formatter):
    """
    只进行预处理。假定样本已通过长度过滤。
    """
    text = formatter(example) 
    tokenized = tokenizer(
        text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length", # 填充到最大长度
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# --- 3. Model Loading (标准 LoRA, 无 4-bit) ---
def get_model_and_tokenizer():
    """Loads the FP16 TinyLlama model with standard LoRA adapters."""
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16, # 以 FP16 加载
        device_map="auto", 
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 
    
    # --- PEFT & LoRA Config ---
    model.gradient_checkpointing_enable() 
    
    lora_config = LoraConfig(
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
    
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters() 
    return model, tokenizer

# --- 4. Main Experiment Logic ---
def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    print(f"--- Loading Base Model & Tokenizer ---")
    base_model, tokenizer = get_model_and_tokenizer()
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Load and Process Datasets ---
    print(f"\n--- Loading and Preprocessing Datasets (This may take a while) ---")
    
    # Task A: HotpotQA
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
    )
    
    hotpot_val_tokenized = hotpot_val.filter(
        lambda x: filter_by_length(x, tokenizer, format_hotpot_qa),
        batched=False,
    ).map(
        lambda x: preprocess(x, tokenizer, format_hotpot_qa),
        batched=False,
    )
    
    print(f"HotpotQA: {len(hotpot_train_tokenized)} train, {len(hotpot_val_tokenized)} val (after filtering)")

    # Task B: MATH
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
    )
    
    math_val_tokenized = math_val.filter(
        lambda x: filter_by_length(x, tokenizer, format_math),
        batched=False,
    ).map(
        lambda x: preprocess(x, tokenizer, format_math),
        batched=False,
    )

    print(f"MATH: {len(math_train_tokenized)} train, {len(math_val_tokenized)} val (after filtering)")

    # --- 【新变更】Experiment 1: Joint Training (Control Group) ---
    print(f"\n--- Starting Experiment 1: Joint Training ---")
    
    joint_model, _ = get_model_and_tokenizer()

    if os.path.exists(os.path.join(JOINT_ADAPTER_PATH, "adapter_model.bin")):
        print(f"--- Found existing Joint adapter. Loading from {JOINT_ADAPTER_PATH} ---")
        joint_model.load_adapter(JOINT_ADAPTER_PATH)
        print("Adapter loaded successfully.")
    
    else:
        print(f"--- No Joint adapter found. Starting Joint Training ---")
        joint_train_dataset = concatenate_datasets([hotpot_train_tokenized, math_train_tokenized]).shuffle(seed=42)
        
        joint_training_args = TrainingArguments(
            output_dir=os.path.join(RESULTS_DIR, "joint_training_temp"), # 临时目录
            per_device_train_batch_size=PER_DEVICE_BS,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            num_train_epochs=JOINT_EPOCHS,
            learning_rate=2e-4,
            fp16=True, 
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
        
        print(f"--- Joint training complete. Saving adapter to {JOINT_ADAPTER_PATH} ---")
        joint_model.save_adapter(JOINT_ADAPTER_PATH)
        print("Adapter saved.")

        del joint_trainer
        torch.cuda.empty_cache()

    # --- 【新变更】Evaluate the "Joint" model (whether trained or loaded) ---
    print("\n--- Evaluating Joint Model ---")
    
    eval_args_joint = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "eval_temp_joint"),
        per_device_eval_batch_size=PER_DEVICE_BS * 2, 
        fp16=True,
        report_to="none",
    )
    
    eval_trainer_joint = Trainer(
        model=joint_model,
        args=eval_args_joint,
        data_collator=data_collator,
    )
    
    eval_hotpot_joint = eval_trainer_joint.evaluate(eval_dataset=hotpot_val_tokenized)
    print(f"  > Joint Model - HotpotQA Val Loss: {eval_hotpot_joint['eval_loss']:.4f}")
    
    eval_math_joint = eval_trainer_joint.evaluate(eval_dataset=math_val_tokenized)
    print(f"  > Joint Model - MATH Val Loss: {eval_math_joint['eval_loss']:.4f}")
    
    del joint_model, eval_trainer_joint, eval_args_joint # 清理显存
    torch.cuda.empty_cache()


    # --- Experiment 2: Sequential Training (Forgetting) ---
    # (此部分逻辑保持不变, 它已经有检查点功能)
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
    
    plt.title(f"Catastrophic Forgetting: HotpotQA -> MATH (Model: {MODEL_NAME} FP16 LoRA)")
    plt.xlabel(f"Training Steps on Task B (MATH) (Total Epochs: {TASK_B_EPOCHS})")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.yscale('log') 
    plt.tight_layout()
    
    plot_filename = os.path.join(RESULTS_DIR, "sequential_forgetting_curve_fp16.png")
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    
    # 在 Colab 中显示图像
    try:
        from google.colab import files
        plt.show()
    except ImportError:
        print("Not in Colab, plot saved to file.")

# 运行主函数
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: This experiment requires a GPU. Check Colab runtime type.")
    else:
        print(f"INFO: Running on GPU. VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        if torch.cuda.get_device_properties(0).total_memory / 1e9 < 11:
            print("WARNING: VRAM is less than 11GB. You may hit OOM errors. Try lowering MAX_SEQ_LENGTH.")
    main()