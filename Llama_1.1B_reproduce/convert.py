import json
import os
import matplotlib.pyplot as plt

# --- 配置 ---
# 确保此路径与 run_experiment.py 中的路径匹配
RESULTS_DIR = "."
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # 用于标题
TASK_B_EPOCHS = 2 # 用于X轴标签 (HotpotQA的Epochs)

# 输入和输出文件名
history_filename = os.path.join(RESULTS_DIR, "forgetting_history_MATH_to_HotpotQA_fp32_7e-5.json")
output_plot_filename = os.path.join(RESULTS_DIR, "./image/replotted_forgetting_curve_MATH_to_HotpotQA_7e-5.png")

def replot_from_json(json_path, output_path):
    """
    从JSON文件加载历史数据并重新生成绘图。
    """
    # --- 1. 加载数据 ---
    if not os.path.exists(json_path):
        print(f"错误：找不到输入文件 {json_path}")
        return

    try:
        with open(json_path, 'r') as f:
            history = json.load(f)
        print(f"成功从 {json_path} 加载历史数据。")
    except Exception as e:
        print(f"错误：无法读取或解析JSON文件：{e}")
        return

    # 验证数据完整性
    if not all(key in history for key in ["steps", "hotpot_loss", "math_loss"]):
        print("错误：JSON文件缺少必要的键 ('steps', 'hotpot_loss', 'math_loss')。")
        return

    # --- 2. 重新绘图 ---
    print("正在生成新绘图...")
    plt.figure(figsize=(12, 6))

    # 根据要求绘制曲线
    # Task A 是 MATH (红色)
    plt.plot(history["steps"], history["math_loss"], 'o-', label="Task A (MATH) Loss", color="red")
    # Task B 是 HotpotQA (蓝色)
    plt.plot(history["steps"], history["hotpot_loss"], 'o-', label="Task B (HotpotQA) Loss", color="blue")

    # 更新标题和标签以反映正确的任务顺序
    plt.title(f"Catastrophic Forgetting: Task A (MATH) -> Task B (HotpotQA) (Model: {MODEL_NAME})")
    plt.xlabel(f"Training Steps on Task B (HotpotQA) (Total Epochs: {TASK_B_EPOCHS})")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()

    # --- 3. 保存绘图 ---
    try:
        plt.savefig(output_path)
        print(f"绘图已成功保存到 {output_path}")
        plt.show()
    except Exception as e:
        print(f"错误：无法保存绘图：{e}")


if __name__ == "__main__":
    # 确保结果目录存在
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"创建了目录：{RESULTS_DIR}")

    replot_from_json(history_filename, output_plot_filename)