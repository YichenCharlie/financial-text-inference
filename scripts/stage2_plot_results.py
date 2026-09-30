from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "figures/stage2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 来源：本次 tiny training 的真实终端输出。
# Epoch 0 表示训练前；这些是同一组 16 条训练样本的评估结果。
epochs = [0, 1, 2, 3]
losses = [0.6137, 0.5833, 0.4365, 0.3533]
accuracies = [0.6250, 0.7500, 0.8750, 0.8750]
correct_counts = [10, 12, 14, 14]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(
    1, 2,
    figsize=(11, 4.8),
    layout="constrained",
)

fig.suptitle(
    "Stage 2 | BERT tiny training — training set only (n=16)",
    fontsize=14,
)

# 左图：loss
axes[0].plot(
    epochs,
    losses,
    marker="o",
    linewidth=2,
    color="#2563eb",
)

for epoch, loss in zip(epochs, losses):
    axes[0].annotate(
        f"{loss:.4f}",
        (epoch, loss),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
    )

axes[0].set(
    title="Loss on the fixed tiny training set",
    xlabel="Epoch (0 = before training)",
    ylabel="Mean cross-entropy loss",
    xticks=epochs,
    xlim=(-0.25, 3.25),
    ylim=(0, 0.75),
)

# 右图：Accuracy
axes[1].plot(
    epochs,
    accuracies,
    marker="o",
    linewidth=2,
    color="#ea580c",
)

for epoch, accuracy, correct in zip(
    epochs, accuracies, correct_counts
):
    axes[1].annotate(
        f"{accuracy:.1%}\n({correct}/16)",
        (epoch, accuracy),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=10,
    )

axes[1].set(
    title="Accuracy on the fixed tiny training set",
    xlabel="Epoch (0 = before training)",
    ylabel="Accuracy",
    xticks=epochs,
    xlim=(-0.25, 3.25),
    ylim=(0, 1.05),
)
axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

for ax in axes:
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)

output_path = OUTPUT_DIR / "tiny_training_curves.png"

fig.savefig(
    output_path,
    dpi=180,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)

print("图片已保存：", output_path)