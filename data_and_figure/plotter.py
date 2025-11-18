import os
import json
import matplotlib.pyplot as plt

# JSON 文件目录
JSON_DIR = "./json/"

# 自动加载目录所有 JSON 文件
json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]

print("Found JSON files:", json_files)

plt.figure(figsize=(12, 7))

for filename in json_files:
    path = os.path.join(JSON_DIR, filename)
    
    # 读取 JSON
    with open(path, "r") as f:
        data = json.load(f)
    
    steps = data["steps"]
    hotpot_loss = data["hotpot_loss"]
    math_loss = data["math_loss"]

    # 从文件名提取实验标签（例如 ratio_0.2.json -> 0.2）
    label_base = filename.replace(".json", "")

    # 画两条曲线（同一张图）
    plt.plot(steps, hotpot_loss, label=f"{label_base} - Hotpot", linestyle="--")
    plt.plot(steps, math_loss, label=f"{label_base} - Math", linestyle="-")

plt.title("Hotpot & Math Validation Loss Across Experiments")
plt.xlabel("Steps")
plt.ylabel("Validation Loss")
plt.yscale("log")     # 如果你想线性坐标可以删掉这行
plt.grid(True)
plt.legend(ncol=2)
plt.tight_layout()

plt.savefig("combined_hotpot_math_loss.png")
print("Saved combined_hotpot_math_loss.png")
plt.show()
