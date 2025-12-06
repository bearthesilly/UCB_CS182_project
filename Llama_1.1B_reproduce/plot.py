import json
import os
import matplotlib.pyplot as plt

# Paths to JSON files
BASE_DIR = r"c:/UCB_CS182_project/Llama_1.1B_reproduce"
RESULT_DIR = os.path.join(BASE_DIR, "result_json")

INTERLEAVE_JSON = os.path.join(RESULT_DIR, "interleave_history_MATH_and_HotpotQA_fp32.json")
JOINT_JSON = os.path.join(RESULT_DIR, "joint_training_history_fp32.json")
FORGETTING_JSON = os.path.join(RESULT_DIR, "forgetting_history_MATH_to_HotpotQA_fp32_2e-4.json")

# Matplotlib global style: enlarge fonts
plt.rcParams.update({
    "font.size": 16,          # base font size
    "axes.titlesize": 20,     # title font size
    "axes.labelsize": 18,     # axis label size
    "legend.fontsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2.5,
    "lines.markersize": 7,
})

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def plot_interleave(data):
    steps = data["steps"]
    hotpot = data["hotpot_loss"]
    math = data["math_loss"]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, hotpot, 'o-', label="HotpotQA Val Loss", color="blue")
    plt.plot(steps, math, 'o-', label="MATH Val Loss", color="red")
    plt.title("Interleaved Training: MATH + HotpotQA")
    plt.xlabel("Training Steps (Total Batches)")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()

def plot_joint(data):
    steps = data["steps"]
    hotpot = data["hotpot_loss"]
    math = data["math_loss"]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, hotpot, 'o-', label="HotpotQA Val Loss", color="blue")
    plt.plot(steps, math, 'o-', label="MATH Val Loss", color="red")
    plt.title("Joint (Mixture) Training: MATH + HotpotQA")
    plt.xlabel("Training Steps")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()

def plot_forgetting(data):
    steps = data["steps"]
    hotpot = data["hotpot_loss"]
    math = data["math_loss"]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, hotpot, 'o-', label="HotpotQA Val Loss (Phase 2)", color="blue")
    plt.plot(steps, math, 'o-', label="MATH Val Loss (Forgetting)", color="red")
    plt.title("Sequential Training (MATH → HotpotQA): Forgetting Curve")
    plt.xlabel("Training Steps")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()

def main():
    # Load data
    interleave_data = load_json(INTERLEAVE_JSON)
    joint_data = load_json(JOINT_JSON)
    forgetting_data = load_json(FORGETTING_JSON)

    # Plot each experiment in its own figure
    plot_interleave(interleave_data)
    plot_joint(joint_data)
    plot_forgetting(forgetting_data)

    # Save figures
    out_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(1)
    plt.savefig(os.path.join(out_dir, "interleave_plot.png"), dpi=200)
    plt.figure(2)
    plt.savefig(os.path.join(out_dir, "joint_plot.png"), dpi=200)
    plt.figure(3)
    plt.savefig(os.path.join(out_dir, "forgetting_plot.png"), dpi=200)

    # Show all figures
    plt.show()

if __name__ == "__main__":
    main()