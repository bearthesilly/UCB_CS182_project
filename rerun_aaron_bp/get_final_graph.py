import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# My RESULTS directory
RESULTS_DIR = "/home/ec2-user/UCB_CS182_project/rerun_aaron_bp/out"

# Alpha values to plot
alphas = [0, 0.05, 0.1, 0.15, 0.2]

# Build filenames 
json_filenames = { alpha: f"forgetting_history_MATH_to_HotpotQA_fp32_alpha_{alpha}.json" for alpha in alphas }

# Storage for parsed data
data = {}

# ---- Load JSON Files ------
for alpha, fname in json_filenames.items():
    path = os.path.join(RESULTS_DIR, fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file missing: {path}")

    with open(path, "r") as f:
        j = json.load(f)
    
    data[alpha] = {
        "steps": j["steps"],
        "hotpot_loss": j["hotpot_loss"],
        "math_loss": j["math_loss"],
    }



alphas = sorted(data.keys())           # ensure sorted order
colors = cm.get_cmap('tab10', len(alphas))   # tab10 gives 10 distinct colors
plt.figure(figsize=(10, 6))

for i,alpha in enumerate(alphas):
    color = colors(i)

    plt.plot(data[alpha]["steps"], data[alpha]["hotpot_loss"], 'o-', label=f"B(HotpotQA) alpha={alpha}", color=color, linewidth=2) #"blue")
    #plt.plot(data[alpha]["steps"], data[alpha]["math_loss"], 'o--', label=f"A (MATH) alpha={alpha}", color=color, linewidth=2) #"red")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TASK_B_EPOCHS = 2
# ---- Plot Hotpot Loss ---- 
plt.title(f"Catastrophic Forgetting: MATH -> HotpotQA (Model: {MODEL_NAME} FP32 LoRA)")
plt.xlabel(f"Training Steps on Task B (HotpotQA) (Total Epochs: {TASK_B_EPOCHS})")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.tight_layout()
plot_filename = os.path.join(RESULTS_DIR, "final_plot_hpqa.png")
plt.savefig(plot_filename)
