import os
import json
import matplotlib.pyplot as plt

# JSON directory
JSON_DIR = "./json/"

# Automatically load all JSON files in the directory
json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]

print("Found JSON files:", json_files)

plt.figure(figsize=(12, 7))

for filename in json_files:
    path = os.path.join(JSON_DIR, filename)
    
    # read JSON files
    with open(path, "r") as f:
        data = json.load(f)
    
    steps = data["steps"]
    hotpot_loss = data["hotpot_loss"]
    math_loss = data["math_loss"]

    # Extract experiment label from filename (e.g., ratio_0.2.json -> 0.2)
    label_base = filename.replace(".json", "")

    # Plot two curves (on the same plot)
    plt.plot(steps, hotpot_loss, label=f"{label_base} - Hotpot", linestyle="--")
    plt.plot(steps, math_loss, label=f"{label_base} - Math", linestyle="-")

plt.title("Hotpot & Math Validation Loss Across Experiments")
plt.xlabel("Steps")
plt.ylabel("Validation Loss")
plt.yscale("log")    
plt.grid(True)
plt.legend(ncol=2)
plt.tight_layout()

plt.savefig("combined_hotpot_math_loss.png")
print("Saved combined_hotpot_math_loss.png")
plt.show()
