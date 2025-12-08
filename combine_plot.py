import json
import os
import matplotlib.pyplot as plt

# --- Configuration ---
# These should match the values used during the experiments
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TASK_B_EPOCHS = 2 # Assuming 2 epochs for Task B in both experiments

# --- File Paths ---
# Adjust these paths if your JSON files are located elsewhere
CODEPARROT_HISTORY_FILE = "./drive/MyDrive/forgetting_history_MATH_to_CodeParrot_bf16.json"
GSM8K_HISTORY_FILE = "./drive/MyDrive/forgetting_history_MATH_to_GSM8K_bf16.json"

# --- Load Data ---
def load_history(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        with open(filepath, 'r') as f:
            history = json.load(f)
        print(f"Successfully loaded history from {filepath}")
        return history
    except Exception as e:
        print(f"Error loading or parsing JSON from {filepath}: {e}")
        return None

code_history = load_history(CODEPARROT_HISTORY_FILE)
gsm8k_history = load_history(GSM8K_HISTORY_FILE)

if code_history is None or gsm8k_history is None:
    print("Cannot proceed without both history files. Please check the file paths.")
else:
    # --- Plot Results ---
    plt.figure(figsize=(15, 8))

    # Plot for MATH -> CodeParrot experiment
    plt.plot(code_history["steps"], code_history["code_loss"], 'o-', label="CodeParrot (Task B) Loss (MATH->Code)", color="blue", alpha=0.7)
    plt.plot(code_history["steps"], code_history["math_loss"], 'x-', label="MATH (Task A) Loss (MATH->Code)", color="red", alpha=0.7)

    # Plot for MATH -> GSM8K experiment
    plt.plot(gsm8k_history["steps"], gsm8k_history["gsm8k_loss"], 'o--', label="GSM8K (Task B) Loss (MATH->GSM8K)", color="green", alpha=0.7)
    plt.plot(gsm8k_history["steps"], gsm8k_history["math_loss"], 'x--', label="MATH (Task A) Loss (MATH->GSM8K)", color="purple", alpha=0.7)

    plt.title(f"Catastrophic Forgetting Comparison (Model: {MODEL_NAME})", fontsize=24)
    plt.xlabel(f"Training Steps on Task B (Total Epochs: {TASK_B_EPOCHS})", fontsize=22)
    plt.ylabel("Validation Loss", fontsize=22)
    plt.legend(fontsize=20) # Adjust legend font size for readability
    plt.grid(True)
    plt.yscale('log') # Use log scale for better visualization of loss changes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    # Save the plot (optional)
    plot_filename = "/content/combined_forgetting_curve.png"
    plt.savefig(plot_filename)
    print(f"Combined plot saved to {plot_filename}")

    plt.show()