"""Plot the recorded Stage 1 results; this script does not train a model.

Default: use the aggregate snapshot transcribed from the original terminal run.
Optional: --predictions PATH recomputes the confusion matrix from the original
JSONL predictions, using fields `label` and `prediction`.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE = "#2563eb", "#e87924"


def scores(cm):
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(2), where=cm.sum(axis=0) != 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(2), where=cm.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(2), where=(precision + recall) != 0)
    return float(tp.sum() / cm.sum()), float(f1.mean()), precision, recall, f1


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(folder / f"{name}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    report = json.loads((ROOT / "results/stage1_baseline/reported_summary.json").read_text(encoding="utf-8"))
    cm = np.array(report["confusion_matrix"], dtype=int)
    if args.predictions:
        cm = np.zeros((2, 2), dtype=int)
        with args.predictions.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                y, pred = row["label"], row["prediction"]
                if y not in (0, 1) or pred not in (0, 1):
                    raise ValueError("Expected integer labels 0 or 1")
                cm[y, pred] += 1
        if cm.sum() == 0:
            raise ValueError("Prediction file is empty")
        print("Source: supplied prediction JSONL (no retraining)")
    else:
        print("Source: reported terminal snapshot (no retraining)")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    folder = ROOT / "figures/stage1_baseline"
    support = cm.sum(axis=1)
    majority_cm = np.zeros((2, 2), dtype=int)
    majority_label = int(np.argmax(report["train_class_counts"]))
    majority_cm[:, majority_label] = support
    a0, f0, *_ = scores(majority_cm)
    a1, f1, precision, recall, per_f1 = scores(cm)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
    x = np.arange(2)
    for offset, values, color, name in [(-.19, [a0, f0], "#94a3b8", "Majority baseline"),
                                       (.19, [a1, f1], BLUE, "TF-IDF + Logistic Regression")]:
        bars = ax.bar(x + offset, values, .36, label=name, color=color)
        ax.bar_label(bars, fmt="%.4f", padding=5)
    ax.set(xticks=x, xticklabels=["Accuracy", "Macro-F1"], ylim=(0, 1), ylabel="Score",
           title=f"Stage 1 | Validation baseline comparison (n = {cm.sum():,})")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    save(fig, folder, "baseline_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9), layout="constrained")
    row_pct = np.divide(cm * 100., support[:, None], out=np.zeros((2, 2)), where=support[:, None] != 0)
    for ax, values, title, vmax in [(axes[0], cm, "Counts", cm.max()),
                                    (axes[1], row_pct, "Within each dataset class (%)", 100)]:
        ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                label = f"{cm[i,j]:,}" if ax is axes[0] else f"{values[i,j]:.1f}%"
                ax.text(j, i, label, ha="center", va="center", fontsize=19,
                        color="white" if values[i,j] > vmax * .52 else "#172554")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-negative", "Negative"],
               yticklabels=["Non-negative", "Negative"], xlabel="Predicted class",
               ylabel="Dataset label", title=title)
        ax.tick_params(length=0)
    fig.suptitle(f"Stage 1 | Confusion matrix: FP = {cm[0,1]:,}, FN = {cm[1,0]:,}", fontsize=15)
    save(fig, folder, "confusion_matrix")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    counts = np.array([report["train_class_counts"], support])
    for i, label in enumerate(["Training", "Validation"]):
        percentages = counts[i] / counts[i].sum() * 100
        left = 0
        for j, (color, name) in enumerate([(BLUE, "Non-negative"), (ORANGE, "Negative")]):
            axes[0].barh(i, percentages[j], left=left, color=color, label=name if i == 0 else None)
            axes[0].text(left + percentages[j]/2, i, f"{percentages[j]:.1f}%\n({counts[i,j]:,})",
                         color="white", ha="center", va="center", fontsize=10)
            left += percentages[j]
    axes[0].set(yticks=[0, 1], yticklabels=["Training", "Validation"], xlim=(0, 100),
                xlabel="Share of company samples (%)", title="Class distribution")
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=2, frameon=False)
    x = np.arange(3)
    for j, (color, name) in enumerate([(BLUE, "Non-negative"), (ORANGE, "Negative")]):
        bars = axes[1].bar(x + (-.19 if j == 0 else .19),
                           [precision[j], recall[j], per_f1[j]], .36, color=color, label=name)
        axes[1].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axes[1].set(xticks=x, xticklabels=["Precision", "Recall", "F1"], ylim=(0, 1.08),
                ylabel="Validation score", title="Performance by class")
    axes[1].grid(axis="y", alpha=.15)
    axes[1].set_axisbelow(True)
    save(fig, folder, "class_distribution_and_metrics")

    toy = json.loads((ROOT / "results/stage1_learning/linear_reported_summary.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    axes[0].plot(toy["logged_steps"], toy["logged_loss"], "o-", color=BLUE)
    axes[0].set(xlabel="Update step (logged every 20 steps)", ylabel="Cross-entropy loss",
                title="Recorded training loss", ylim=(0, .9))
    axes[0].grid(alpha=.2)
    w, b = np.array(toy["weight_after"]), np.array(toy["bias_after"])
    boundary = (b[1] - b[0]) / (w[0] - w[1])
    xx = np.linspace(-3.2, 3.2, 250)
    for k, color in enumerate([BLUE, ORANGE]):
        axes[1].plot(xx, w[k]*xx+b[k], label=f"Class {k} score", color=color)
    axes[1].axvline(boundary, color="#64748b", linestyle="--", label=f"Boundary ≈ {boundary:.3f}")
    axes[1].set(xlabel="Input x", ylabel="Logit (not a probability)", title="Learned scores from rounded parameters")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(alpha=.15)
    fig.suptitle("Learning exercise only | nn.Linear(1, 2), 4 synthetic samples", fontsize=14)
    save(fig, ROOT / "figures/stage1_learning", "linear_learning")


if __name__ == "__main__":
    main()
