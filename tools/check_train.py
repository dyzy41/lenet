import json
import argparse
import matplotlib.pyplot as plt
import os

def plot_loss_vs_step(log_path, output_loss_image_path, output_miou_macc_image_path):
    steps_loss = []
    losses = []
    steps_miou_macc = []
    mIoU = []
    mAcc = []
    
    # 逐行读取 JSON 文件
    with open(os.path.join(log_path, 'vis_data/scalars.json'), 'r') as f:
        for line in f:
            entry = json.loads(line.strip())  # 逐行解析 JSON
            if "loss" in entry:  # 仅处理包含 'loss' 字段的行
                steps_loss.append(entry["step"])
                losses.append(entry["loss"])
            if "mIoU" in entry:  # 处理验证结果行
                steps_miou_macc.append(entry["step"])
                mIoU.append(entry["mIoU"])
                mAcc.append(entry["mAcc"])

    # 绘制 Loss vs Step 图
    plt.figure(figsize=(10, 6))
    plt.plot(steps_loss, losses, linestyle='-', linewidth=0.8, color='r', label="Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss vs Step")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(log_path, output_loss_image_path))
    plt.show()
    
    # 绘制 mIoU 和 mAcc vs Step 图
    plt.figure(figsize=(10, 6))
    plt.plot(steps_miou_macc, mIoU, linestyle='-', linewidth=0.8, color='b', label="mIoU")
    plt.plot(steps_miou_macc, mAcc, linestyle='-', linewidth=0.8, color='g', label="mAcc")
    plt.xlabel("Step")
    plt.ylabel("Metrics")
    plt.title("mIoU and mAcc vs Step")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(log_path, output_miou_macc_image_path))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Loss, mIoU, and mAcc from JSON file.")
    parser.add_argument("log_path", type=str, help="Path to the JSON file.")
    parser.add_argument("--output_loss_image_path", type=str, default="0loss.png", help="Output path for the Loss vs Step plot.")
    parser.add_argument("--output_miou_macc_image_path", type=str, default="0miou_macc.png", help="Output path for the mIoU and mAcc vs Step plot.")
    
    args = parser.parse_args()
    
    plot_loss_vs_step(args.log_path, args.output_loss_image_path, args.output_miou_macc_image_path)

