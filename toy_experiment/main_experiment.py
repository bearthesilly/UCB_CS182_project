# main_experiment.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import warnings

# --- 1. Configuration ---
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

CONFIG = {
    # Model: DistilBERT is small and VRAM-friendly (fits in 8GB)
    "model_name": "./offline_assets/distilbert-base-uncased",
    
    # Dataset: AG News has 4 distinct classes we can split
    "dataset_name": "./offline_assets/ag_news",
    
    # Task Split:
    # Task A: World (class 0) and Sports (class 1)
    # Task B: Business (class 2) and Sci/Tech (class 3)
    "task_a_labels": [0, 1],
    "task_b_labels": [2, 3],
    
    # Training Hyperparameters
    "batch_size": 16, # Safe batch size for 8GB VRAM
    "num_epochs_a": 3,  # Epochs to master Task A
    "num_epochs_b": 3,  # Epochs to learn Task B (and forget A)
    "num_epochs_joint": 3, # Epochs for the control group
    "learning_rate": 2e-5,
    "max_seq_length": 128, # Keep sequences short to save VRAM
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# --- 2. Data Preparation ---

def get_dataloaders(tokenizer):
    """
    Loads and preprocesses the AG News dataset, splitting it into
    Task A, Task B, and a Joint (control) task.
    """
    print("--- Loading and Preparing Data ---")
    
    # Load dataset
    raw_datasets = load_from_disk(CONFIG["dataset_name"])

    # Rename label column for consistency
    if "label" not in raw_datasets["train"].column_names:
        raw_datasets = raw_datasets.rename_column("labels", "label")

    # Tokenization function
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=CONFIG["max_seq_length"],
        )

    # Tokenize the entire dataset
    tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # --- Create Task-Specific Datasets ---
    
    # Task A (World, Sports)
    task_a_train = tokenized_datasets["train"].filter(
        lambda x: x["labels"] in CONFIG["task_a_labels"]
    )
    task_a_val = tokenized_datasets["test"].filter(
        lambda x: x["labels"] in CONFIG["task_a_labels"]
    )
    
    # Task B (Business, Sci/Tech)
    task_b_train = tokenized_datasets["train"].filter(
        lambda x: x["labels"] in CONFIG["task_b_labels"]
    )
    task_b_val = tokenized_datasets["test"].filter(
        lambda x: x["labels"] in CONFIG["task_b_labels"]
    )

    # Task Joint (Control Group - All 4 classes)
    task_joint_train = tokenized_datasets["train"]
    task_joint_val = tokenized_datasets["test"] # Use full test set for eval

    # Create DataLoaders
    # We will *always* evaluate on both A and B validation sets
    loader_a_val = DataLoader(task_a_val, batch_size=CONFIG["batch_size"])
    loader_b_val = DataLoader(task_b_val, batch_size=CONFIG["batch_size"])

    loaders = {
        "a_train": DataLoader(task_a_train, batch_size=CONFIG["batch_size"], shuffle=True),
        "b_train": DataLoader(task_b_train, batch_size=CONFIG["batch_size"], shuffle=True),
        "joint_train": DataLoader(task_joint_train, batch_size=CONFIG["batch_size"], shuffle=True),
        "a_val": loader_a_val,
        "b_val": loader_b_val,
        # We also create a full validation set for the joint model's primary metric
        "joint_val": DataLoader(task_joint_val, batch_size=CONFIG["batch_size"])
    }
    
    print(f"Task A (Train/Val): {len(task_a_train)} / {len(task_a_val)}")
    print(f"Task B (Train/Val): {len(task_b_train)} / {len(task_b_val)}")
    print(f"Task Joint (Train/Val): {len(task_joint_train)} / {len(task_joint_val)}")
    
    return loaders


# --- 3. Model & Helper Functions ---

def get_model():
    """
    Initializes a fresh DistilBERT model for 4-class classification.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=4, # We always use a 4-label head
    ).to(CONFIG["device"])
    return model

@torch.no_grad()
def evaluate(model, loader_a_val, loader_b_val, device):
    """
    Evaluates the model's performance on *both* Task A and Task B
    validation sets to track learning and forgetting.
    """
    model.eval()
    
    # --- Evaluate Task A (Forgetting) ---
    all_preds_a, all_labels_a = [], []
    total_loss_a = 0
    
    for batch in loader_a_val:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        
        # We use outputs.loss because the model computes loss internally
        # as long as we provide 'labels'.
        total_loss_a += outputs.loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        
        all_preds_a.extend(preds.cpu().numpy())
        all_labels_a.extend(batch["labels"].cpu().numpy())

    acc_a = accuracy_score(all_labels_a, all_preds_a)
    loss_a = total_loss_a / len(loader_a_val)
    
    # --- Evaluate Task B (Learning) ---
    all_preds_b, all_labels_b = [], []
    total_loss_b = 0

    for batch in loader_b_val:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        
        total_loss_b += outputs.loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        
        all_preds_b.extend(preds.cpu().numpy())
        all_labels_b.extend(batch["labels"].cpu().numpy())
        
    acc_b = accuracy_score(all_labels_b, all_preds_b)
    loss_b = total_loss_b / len(loader_b_val)
    
    return {
        "acc_a": acc_a, "loss_a": loss_a,
        "acc_b": acc_b, "loss_b": loss_b,
    }

def train_one_epoch(model, train_loader, optimizer, scheduler, device):
    """
    A single training epoch.
    """
    model.train()
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({"loss": loss.item()})

# --- 4. Experiment Execution ---

def run_joint_training(loaders):
    """
    Experiment 1: Control Group (Joint Training)
    Train on Task A and B simultaneously. This is the 'ideal' scenario.
    """
    print("\n--- Experiment 1: Running Joint Training (Control) ---")
    
    model = get_model()
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    
    num_training_steps = CONFIG["num_epochs_joint"] * len(loaders["joint_train"])
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    history = []

    for epoch in range(CONFIG["num_epochs_joint"]):
        print(f"Joint Epoch {epoch+1}/{CONFIG['num_epochs_joint']}")
        
        # Train on all 4 classes
        train_one_epoch(model, loaders["joint_train"], optimizer, scheduler, CONFIG["device"])
        
        # Evaluate on both A and B validation sets
        metrics = evaluate(model, loaders["a_val"], loaders["b_val"], CONFIG["device"])
        
        print(f"  > Acc A (World/Sports): {metrics['acc_a']:.4f} (Loss: {metrics['loss_a']:.4f})")
        print(f"  > Acc B (Biz/Sci): {metrics['acc_b']:.4f} (Loss: {metrics['loss_b']:.4f})")
        
        metrics['epoch'] = epoch + 1
        history.append(metrics)
        
    return history, model

def run_sequential_training(loaders):
    """
    Experiment 2: Catastrophic Forgetting
    Phase 1: Train *only* on Task A.
    Phase 2: Train *only* on Task B (using the model from Phase 1).
    """
    print("\n--- Experiment 2: Running Sequential Training (Forgetting) ---")
    
    model = get_model()
    history = []
    
    # --- Phase 1: Train on Task A (World/Sports) ---
    print("  --- Phase 1: Training on Task A (World/Sports) ---")
    
    optimizer_a = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    num_training_steps_a = CONFIG["num_epochs_a"] * len(loaders["a_train"])
    scheduler_a = get_scheduler(
        "linear",
        optimizer_a,
        num_warmup_steps=0,
        num_training_steps=num_training_steps_a,
    )

    for epoch in range(CONFIG["num_epochs_a"]):
        print(f"Sequential Epoch (Task A) {epoch+1}/{CONFIG['num_epochs_a']}")
        
        # Train *only* on Task A
        train_one_epoch(model, loaders["a_train"], optimizer_a, scheduler_a, CONFIG["device"])
        
        # Evaluate on both A and B validation sets
        metrics = evaluate(model, loaders["a_val"], loaders["b_val"], CONFIG["device"])
        
        print(f"  > Acc A (World/Sports): {metrics['acc_a']:.4f} (Loss: {metrics['loss_a']:.4f})")
        print(f"  > Acc B (Biz/Sci): {metrics['acc_b']:.4f} (Loss: {metrics['loss_b']:.4f})")
        
        metrics['epoch'] = epoch + 1
        history.append(metrics)

    print(f"\n  --- Finished Phase 1. Model is now an 'expert' on Task A. ---\n")

    # --- Phase 2: Train on Task B (Business/Sci) ---
    print("  --- Phase 2: Training on Task B (Business/Sci) ---")
    print("  *** WATCH TASK A ACCURACY AND LOSS HERE ***")

    # Re-initialize optimizer for the new task.
    # We continue using the *same model*.
    optimizer_b = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    num_training_steps_b = CONFIG["num_epochs_b"] * len(loaders["b_train"])
    scheduler_b = get_scheduler(
        "linear",
        optimizer_b,
        num_warmup_steps=0,
        num_training_steps=num_training_steps_b,
    )

    for epoch in range(CONFIG["num_epochs_b"]):
        print(f"Sequential Epoch (Task B) {epoch+1}/{CONFIG['num_epochs_b']}")
        
        # Train *only* on Task B
        train_one_epoch(model, loaders["b_train"], optimizer_b, scheduler_b, CONFIG["device"])
        
        # Evaluate on both A and B validation sets
        metrics = evaluate(model, loaders["a_val"], loaders["b_val"], CONFIG["device"])
        
        # This is the key moment. Acc A should drop, Loss A should rise.
        print(f"  > Acc A (World/Sports): {metrics['acc_a']:.4f} (Loss: {metrics['loss_a']:.4f}) <-- FORGETTING!")
        print(f"  > Acc B (Biz/Sci): {metrics['acc_b']:.4f} (Loss: {metrics['loss_b']:.4f}) <-- LEARNING!")
        
        # We offset the epoch number for continuous plotting
        metrics['epoch'] = epoch + 1 + CONFIG["num_epochs_a"]
        history.append(metrics)

    return history, model

# --- 5. Plotting ---

def plot_results(joint_history, seq_history):
    """
    Generates the 4-panel plot showing the difference between
    Joint and Sequential (CF) training.
    """
    print("\n--- Generating Plots ---")
    
    # Helper to extract data from history lists
    def get_metrics(history):
        epochs = [h['epoch'] for h in history]
        acc_a = [h['acc_a'] for h in history]
        loss_a = [h['loss_a'] for h in history]
        acc_b = [h['acc_b'] for h in history]
        loss_b = [h['loss_b'] for h in history]
        return epochs, acc_a, loss_a, acc_b, loss_b

    j_epochs, j_acc_a, j_loss_a, j_acc_b, j_loss_b = get_metrics(joint_history)
    s_epochs, s_acc_a, s_loss_a, s_acc_b, s_loss_b = get_metrics(seq_history)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Catastrophic Forgetting: Joint vs. Sequential Training on AG News", fontsize=16)

    # --- Plot 1: Task A Accuracy (The Forgetting Curve) ---
    ax1 = axes[0, 0]
    ax1.plot(j_epochs, j_acc_a, 'o-', label="Joint Training (Control)")
    ax1.plot(s_epochs, s_acc_a, 'o-', label="Sequential Training (CF)")
    # Add a vertical line to show where Task B training begins
    ax1.axvline(x=CONFIG["num_epochs_a"], color='r', linestyle='--', label="Task B Starts")
    ax1.set_title("Task A (World/Sports) Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    # --- Plot 2: Task B Accuracy (The Learning Curve) ---
    ax2 = axes[0, 1]
    ax2.plot(j_epochs, j_acc_b, 'o-', label="Joint Training (Control)")
    ax2.plot(s_epochs, s_acc_b, 'o-', label="Sequential Training (CF)")
    ax2.axvline(x=CONFIG["num_epochs_a"], color='r', linestyle='--', label="Task B Starts")
    ax2.set_title("Task B (Business/Sci) Accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    # --- Plot 3: Task A Loss (The Forgetting Curve) ---
    ax3 = axes[1, 0]
    ax3.plot(j_epochs, j_loss_a, 'o-', label="Joint Training (Control)")
    ax3.plot(s_epochs, s_loss_a, 'o-', label="Sequential Training (CF)")
    ax3.axvline(x=CONFIG["num_epochs_a"], color='r', linestyle='--', label="Task B Starts")
    ax3.set_title("Task A (World/Sports) Loss")
    ax3.set_ylabel("Loss")
    ax3.set_xlabel("Epoch")
    ax3.legend()
    ax3.grid(True)
    ax3.set_yscale('log') # Loss often explodes, log scale is better

    # --- Plot 4: Task B Loss (The Learning Curve) ---
    ax4 = axes[1, 1]
    ax4.plot(j_epochs, j_loss_b, 'o-', label="Joint Training (Control)")
    ax4.plot(s_epochs, s_loss_b, 'o-', label="Sequential Training (CF)")
    ax4.axvline(x=CONFIG["num_epochs_a"], color='r', linestyle='--', label="Task B Starts")
    ax4.set_title("Task B (Business/Sci) Loss")
    ax4.set_ylabel("Loss")
    ax4.set_xlabel("Epoch")
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("catastrophic_forgetting_results.png")
    print("\nResults saved to 'catastrophic_forgetting_results.png'")
    plt.show()


# --- 6. Main Execution ---

if __name__ == "__main__":
    
    if CONFIG["device"] == "cpu":
        print("WARNING: Running on CPU. This will be very slow.")
    else:
        print(f"INFO: Running on {CONFIG['device']} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    
    # Get all dataloaders
    all_loaders = get_dataloaders(tokenizer)
    
    # Run Experiment 1
    joint_history, model_joint = run_joint_training(all_loaders)
    
    # Run Experiment 2
    seq_history, model_seq = run_sequential_training(all_loaders)
    
    # Plot the results
    plot_results(joint_history, seq_history)