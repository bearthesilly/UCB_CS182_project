import json
import os
import matplotlib.pyplot as plt

RESULTS_DIR = "."
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
TASK_B_EPOCHS = 2 

history_filename = os.path.join(RESULTS_DIR, "forgetting_history_MATH_to_HotpotQA_fp32_7e-5.json")
output_plot_filename = os.path.join(RESULTS_DIR, "./image/replotted_forgetting_curve_MATH_to_HotpotQA_7e-5.png")

def replot_from_json(json_path, output_path):

    try:
        with open(json_path, 'r') as f:
            history = json.load(f)
    except Exception as e:
        return

    plt.figure(figsize=(12, 6))

    plt.plot(history["steps"], history["math_loss"], 'o-', label="Task A (MATH) Loss", color="red")
    plt.plot(history["steps"], history["hotpot_loss"], 'o-', label="Task B (HotpotQA) Loss", color="blue")

    plt.title(f"Catastrophic Forgetting: Task A (MATH) -> Task B (HotpotQA) (Model: {MODEL_NAME})")
    plt.xlabel(f"Training Steps on Task B (HotpotQA) (Total Epochs: {TASK_B_EPOCHS})")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()

    try:
        plt.savefig(output_path)
        print(f"Figure saved to {output_path}")
        plt.show()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    replot_from_json(history_filename, output_plot_filename)