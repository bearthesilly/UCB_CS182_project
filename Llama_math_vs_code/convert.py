# convert.py
"""
Replot forgetting curves for math vs coding experiment from result JSON.
"""
import json
import os
import matplotlib.pyplot as plt

RESULTS_DIR = "./result_json/"
history_filename = os.path.join(RESULTS_DIR, "forgetting_history_math_vs_code.json")
output_plot_filename = os.path.join("./image/", "replotted_forgetting_curve_math_vs_code.png")

os.makedirs("./image/", exist_ok=True)


def replot_from_json(json_path, output_path):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
    with open(json_path, 'r') as f:
        history = json.load(f)

    # Improved plot: bar chart with log scale, clear labels
    plt.figure(figsize=(10, 6))
    bars = ["Math Loss (Before Coding)", "Math Loss (After Coding)", "Coding Loss"]
    values = [history["math_loss_before"], history["math_loss_after"], history["coding_loss"]]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    plt.bar(bars, values, color=colors)
    plt.ylabel("Validation Loss (log scale)")
    plt.yscale('log')
    plt.title("Catastrophic Forgetting: TinyLlama on Math vs Coding (mbpp)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    replot_from_json(history_filename, output_plot_filename)
