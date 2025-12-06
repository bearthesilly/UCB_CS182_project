#!/usr/bin/env python3

import os, json, warnings, logging
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, TrainerCallback
)
from peft import LoraConfig, get_peft_model

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HOTPOT_DATASET_NAME = "hotpot_qa"
HOTPOT_DATASET_CONFIG = "distractor"
MATH_DATASET_NAME = "qwedsacf/competition_math"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TASK_A_EPOCHS = 8
TASK_B_EPOCHS = 8
N_TRAIN_EXAMPLES = 4000
N_VAL_EXAMPLES = 400
MAX_SEQ_LENGTH = 512
PER_DEVICE_BS = 32
GRAD_ACC_STEPS = 4
EVAL_EVERY = 10            # evaluate forgetting every N steps
SMOOTHING_COEFFS = [0, 0.05, 0.1, 0.15, 0.2]

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
logging.basicConfig(
    filename=os.path.join(RESULTS_DIR, "logger.txt"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Data formatting functions
# ---------------------------------------------------------------------
def format_hotpot(example):
    ctx = " ".join(["".join(s) for s in example["context"]["sentences"]])
    return f"<s>[INST] Using this context answer the question.\nContext: {ctx}\n\nQuestion: {example['question']} [/INST] Answer: {example['answer']}</s>"

def format_math(example):
    return f"<s>[INST] Solve this math problem. Show your work.\nProblem: {example['problem']} [/INST] Solution: {example['solution']}</s>"

def filter_by_length(example, tokenizer, formatter):
    return len(tokenizer(formatter(example), max_length=MAX_SEQ_LENGTH + 1).input_ids) <= MAX_SEQ_LENGTH

def preprocess(example, tokenizer, formatter):
    text = formatter(example)
    t = tokenizer(text, max_length=MAX_SEQ_LENGTH, truncation=True, padding="max_length")
    labels = t["input_ids"].copy()

    inst_token = tokenizer.convert_tokens_to_ids("]")
    # ignore prompt part
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] == inst_token:
            labels[: i + 1] = [-100] * (i + 1)
            break

    t["labels"] = labels
    return t

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
def get_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

def get_lora_cfg():
    return LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

# ---------------------------------------------------------------------
# Smoothed loss Trainer (α-loss)
# ---------------------------------------------------------------------
class SmoothedTrainer(Trainer):
    def __init__(self, *a, alpha=0.0, **kw):
        super().__init__(*a, **kw)
        self.alpha = alpha
        self.prev_loss = None

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = super().compute_loss(model, inputs, return_outputs=True)
        loss, outputs = out
        if self.prev_loss is None:
            self.prev_loss = loss.detach()
        smooth = (1 - self.alpha) * loss + self.alpha * self.prev_loss
        self.prev_loss = loss.detach()
        return (smooth, outputs) if return_outputs else smooth

# ---------------------------------------------------------------------
# Forgetting Callback (no recursion, no stopping training)
# ---------------------------------------------------------------------
class ForgettingCallback(TrainerCallback):
    def __init__(self, math_dl, hotpot_dl, history, eval_every):
        self.math_dl = math_dl
        self.hotpot_dl = hotpot_dl
        self.history = history
        self.eval_every = eval_every
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    @torch.no_grad()
    def eval_loss(self, dataloader):
        model = self.trainer.model
        model.eval()
        device = next(model.parameters()).device
        total, count = 0, 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids","attention_mask","labels"]}
            loss = model(**batch).loss
            total += loss.item(); count += 1
        model.train()
        return total / count

    def on_step_end(self, args, state, control, **kw):
        if state.global_step == 0 or state.global_step % self.eval_every != 0:
            return
        step = state.global_step
        logger.info(f"\n[Forgetting] step {step} — running eval")
        print(f"\n[Forgetting] step {step} — running eval")
        hotpot_loss = self.eval_loss(self.hotpot_dl)
        math_loss = self.eval_loss(self.math_dl)
        logger.info(f"  Hotpot val = {hotpot_loss:.4f}")
        logger.info(f"  Math   val = {math_loss:.4f}")
        print(f"  Hotpot val = {hotpot_loss:.4f}")
        print(f"  Math   val = {math_loss:.4f}")
        self.history["steps"].append(step)
        self.history["hotpot_loss"].append(hotpot_loss)
        self.history["math_loss"].append(math_loss)

# ---------------------------------------------------------------------
# Experiment logic
# ---------------------------------------------------------------------
def run(alpha):
    logger.info(f"\n===== SMOOTHING α = {alpha} =====")
    print(f"\n===== SMOOTHING α = {alpha} =====")
    model, tokenizer = get_model_and_tokenizer()
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # load + tokenize datasets
    raw_hotpot = load_dataset(HOTPOT_DATASET_NAME, HOTPOT_DATASET_CONFIG)
    hotpot_train = raw_hotpot["train"].shuffle(seed=42).select(range(N_TRAIN_EXAMPLES))
    hotpot_val = raw_hotpot["validation"].shuffle(seed=42).select(range(N_VAL_EXAMPLES))

    raw_math = load_dataset(MATH_DATASET_NAME)
    subset = raw_math["train"].shuffle(seed=42).select(range(N_TRAIN_EXAMPLES + N_VAL_EXAMPLES))
    splits = subset.train_test_split(test_size=N_VAL_EXAMPLES / (N_TRAIN_EXAMPLES + N_VAL_EXAMPLES), seed=42)
    math_train, math_val = splits["train"], splits["test"]

    hotpot_train = hotpot_train.filter(lambda x: filter_by_length(x, tokenizer, format_hotpot)) \
                               .map(lambda x: preprocess(x, tokenizer, format_hotpot)).filter(lambda x: len(x)>0)
    hotpot_val   = hotpot_val.filter(lambda x: filter_by_length(x, tokenizer, format_hotpot)) \
                             .map(lambda x: preprocess(x, tokenizer, format_hotpot)).filter(lambda x: len(x)>0)
    math_train   = math_train.filter(lambda x: filter_by_length(x, tokenizer, format_math)) \
                             .map(lambda x: preprocess(x, tokenizer, format_math)).filter(lambda x: len(x)>0)
    math_val     = math_val.filter(lambda x: filter_by_length(x, tokenizer, format_math)) \
                           .map(lambda x: preprocess(x, tokenizer, format_math)).filter(lambda x: len(x)>0)

    # Phase 1 (Math)
    model = get_peft_model(model, get_lora_cfg())
    argsA = TrainingArguments(
        output_dir=f"{RESULTS_DIR}/phaseA",
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_train_epochs=TASK_A_EPOCHS,
        learning_rate=2e-4,
        save_strategy="no",
        report_to="none"
    )
    trainerA = SmoothedTrainer(model=model, args=argsA, train_dataset=math_train,
                               eval_dataset=math_val, data_collator=data_collator, alpha=alpha)
    trainerA.train()
    del trainerA; torch.cuda.empty_cache()

    # Eval after Phase 1
    eval_tr = Trainer(model=model, args=TrainingArguments(output_dir=f"{RESULTS_DIR}/tmp", report_to="none"),
                      data_collator=data_collator)
    start_hotpot = eval_tr.evaluate(eval_dataset=hotpot_val)["eval_loss"]
    start_math   = eval_tr.evaluate(eval_dataset=math_val)["eval_loss"]
    logger.info(f"Phase1 Eval — Hotpot={start_hotpot:.4f}, Math={start_math:.4f}")
    print(f"Phase1 Eval — Hotpot={start_hotpot:.4f}, Math={start_math:.4f}")

    # Phase 2 (Hotpot with forgetting tracking)
    history = {"steps": [], "hotpot_loss": [], "math_loss": []}
    math_dl = DataLoader(math_val, batch_size=PER_DEVICE_BS)
    hotpot_dl = DataLoader(hotpot_val, batch_size=PER_DEVICE_BS)
    cb = ForgettingCallback(math_dl, hotpot_dl, history, eval_every=EVAL_EVERY)

    argsB = TrainingArguments(
        output_dir=f"{RESULTS_DIR}/phaseB",
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_train_epochs=TASK_B_EPOCHS,
        learning_rate=7e-5,
        save_strategy="no",
        report_to="none",
        logging_steps=20,
    )
    trainerB = SmoothedTrainer(model=model, args=argsB, train_dataset=hotpot_train,
                               eval_dataset=hotpot_val, data_collator=data_collator,
                               callbacks=[cb], alpha=alpha)
    cb.set_trainer(trainerB)
    trainerB.train()

    # save plot + json
    fname = f"{RESULTS_DIR}/forgetting_alpha_{alpha}.json"
    with open(fname, "w") as f: json.dump(history, f, indent=4)
    logger.info(f"Saved history → {fname}")
    print(f"Saved history → {fname}")

    plt.figure(figsize=(10,5))
    plt.plot(history["steps"], history["hotpot_loss"], label="Hotpot")
    plt.plot(history["steps"], history["math_loss"], label="Math")
    plt.yscale("log")
    plt.legend(); plt.grid()
    plt.savefig(f"{RESULTS_DIR}/forgetting_alpha_{alpha}.png")
    logger.info("Plot saved.")
    print("Plot saved.")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    assert torch.cuda.is_available(), "GPU required"
    logger.info(f"GPU: {torch.cuda.get_device_properties(0).name}")
    print(f"GPU: {torch.cuda.get_device_properties(0).name}")

    for alpha in SMOOTHING_COEFFS:
        run(alpha)

