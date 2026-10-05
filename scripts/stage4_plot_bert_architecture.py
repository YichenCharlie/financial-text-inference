from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "figures/stage4_bert"


# 使用常见字体；PDF 保留可编辑文字。
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})

COLORS = {
    "ink": "#263445",
    "muted": "#647184",
    "line": "#66758A",
    "blue": "#E9F0F8",
    "purple": "#F0ECF7",
    "green": "#E9F3ED",
    "orange": "#FBF0E3",
    "gray": "#F3F5F7",
}


def setup_panel(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis("off")
    ax.text(
        0.2, 14.65, title,
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
    )


def box(ax, x, y, w, h, title, subtitle=None, fill="gray"):
    """x、y 为框左下角坐标，返回常用连接点。"""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=COLORS[fill],
        edgecolor=COLORS["line"],
        linewidth=1.0,
    )
    ax.add_patch(patch)

    if subtitle:
        title_y = y + h * 0.65
        subtitle_y = y + h * 0.28
        ax.text(
            x + w / 2, title_y, title,
            ha="center", va="center",
            fontsize=10, fontweight="medium",
            color=COLORS["ink"],
        )
        ax.text(
            x + w / 2, subtitle_y, subtitle,
            ha="center", va="center",
            fontsize=8.6,
            color=COLORS["muted"],
        )
    else:
        ax.text(
            x + w / 2, y + h / 2, title,
            ha="center", va="center",
            fontsize=10,
            color=COLORS["ink"],
        )

    return {
        "top": (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
        "left": (x, y + h / 2),
        "right": (x + w, y + h / 2),
    }


def arrow(ax, start, end, dashed=False):
    patch = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.05,
        color=COLORS["line"],
        linestyle="--" if dashed else "-",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)


def connect(ax, upper, lower):
    arrow(ax, upper["bottom"], lower["top"])


def residual(ax, start, end, route_x):
    """残差分支：先向侧面，再向下，最后带箭头进入加法节点。"""
    ax.plot(
        [start[0], route_x, route_x],
        [start[1], start[1], end[1]],
        color=COLORS["line"],
        linewidth=1.05,
    )
    arrow(ax, (route_x, end[1]), end)


def addition(ax, center):
    cx, cy = center
    ax.scatter(
        [cx], [cy],
        s=350,
        facecolor="white",
        edgecolor=COLORS["line"],
        linewidth=1.1,
        zorder=4,
    )
    ax.text(
        cx, cy, "+",
        ha="center", va="center",
        fontsize=15, color=COLORS["ink"],
        zorder=5,
    )


def draw_overview(ax):
    setup_panel(ax, "(a) BERT classification pipeline")

    nodes = [
        box(
            ax, 1.0, 12.5, 8.0, 1.1,
            "Target company + news",
            "Paired tokenization; maximum length = 512",
        ),
        box(
            ax, 1.0, 10.55, 8.0, 1.25,
            "Embedding",
            "Token + position + segment; LayerNorm",
            fill="blue",
        ),
        box(
            ax, 1.0, 8.45, 8.0, 1.25,
            "12 Encoder layers",
            "Independent parameters in each layer",
            fill="purple",
        ),
        box(
            ax, 1.0, 6.65, 8.0, 1.0,
            "Select final [CLS] representation",
            r"$[B,\ 768]$",
            fill="blue",
        ),
        box(
            ax, 1.0, 4.85, 8.0, 1.0,
            "Pooler",
            r"Linear $768 \rightarrow 768$ + Tanh",
            fill="green",
        ),
        box(
            ax, 1.0, 3.05, 8.0, 1.0,
            "Classifier",
            r"Linear $768 \rightarrow 2$",
            fill="orange",
        ),
        box(
            ax, 1.0, 1.25, 8.0, 1.0,
            "Softmax probabilities",
            "non-negative (0) / negative (1)",
        ),
    ]

    for upper, lower in zip(nodes, nodes[1:]):
        connect(ax, upper, lower)

    ax.text(
        5.0, 10.12, r"$X \in \mathbb{R}^{B \times L \times 768}$",
        ha="center", va="center",
        fontsize=9, color=COLORS["muted"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
    )


def draw_encoder(ax):
    setup_panel(ax, "(b) One Encoder layer: post-LN")

    x = box(
        ax, 2.0, 12.6, 6.0, 0.9,
        r"Input $X$: $[B,\ L,\ 768]$",
        fill="blue",
    )
    attention = box(
        ax, 2.0, 10.75, 6.0, 1.15,
        "Multi-head attention",
        "12 heads + output projection",
        fill="purple",
    )
    connect(ax, x, attention)

    add1 = (5.0, 10.0)
    arrow(ax, attention["bottom"], (5.0, 10.20))
    residual(ax, (5.0, 12.25), (4.76, 10.0), route_x=0.8)
    addition(ax, add1)

    norm1 = box(
        ax, 2.0, 8.65, 6.0, 0.85,
        "LayerNorm",
        fill="green",
    )
    arrow(ax, (5.0, 9.78), norm1["top"])

    ax.text(
        5.25, 8.22, r"$H$",
        fontsize=10, color=COLORS["muted"],
        va="center",
    )

    linear1 = box(
        ax, 2.0, 6.95, 6.0, 0.85,
        r"Linear: $768 \rightarrow 3072$",
        fill="orange",
    )
    gelu = box(
        ax, 2.0, 5.65, 6.0, 0.85,
        "GELU",
        fill="orange",
    )
    linear2 = box(
        ax, 2.0, 4.35, 6.0, 0.85,
        r"Linear: $3072 \rightarrow 768$",
        fill="orange",
    )

    connect(ax, norm1, linear1)
    connect(ax, linear1, gelu)
    connect(ax, gelu, linear2)

    # 标记 FFN 范围。
    ax.plot(
        [8.45, 8.7, 8.7, 8.45],
        [7.8, 7.8, 4.35, 4.35],
        color=COLORS["muted"],
        linewidth=1.0,
    )
    ax.text(
        9.05, 6.08, "FFN",
        rotation=90,
        ha="center", va="center",
        fontsize=9, color=COLORS["muted"],
    )

    add2 = (5.0, 3.55)
    arrow(ax, linear2["bottom"], (5.0, 3.77))
    residual(ax, (5.0, 8.22), (4.76, 3.55), route_x=1.2)
    addition(ax, add2)

    norm2 = box(
        ax, 2.0, 2.15, 6.0, 0.85,
        "LayerNorm",
        fill="green",
    )
    arrow(ax, (5.0, 3.33), norm2["top"])

    output = box(
        ax, 2.0, 0.65, 6.0, 0.85,
        r"Output: $[B,\ L,\ 768]$",
        fill="blue",
    )
    connect(ax, norm2, output)


def draw_attention(ax):
    setup_panel(ax, "(c) Inside multi-head attention")

    x = box(
        ax, 1.0, 12.6, 8.0, 0.9,
        r"Input $X$: $[B,\ L,\ 768]$",
        fill="blue",
    )

    qkv = []
    for left, name in [(0.6, "Q"), (3.7, "K"), (6.8, "V")]:
        node = box(
            ax, left, 10.65, 2.6, 1.1,
            f"{name} projection",
            "768 → 768",
            fill="purple",
        )
        qkv.append(node)
        arrow(ax, x["bottom"], node["top"])

    split = box(
        ax, 1.0, 8.7, 8.0, 1.1,
        "Split each Q, K, V into 12 heads",
        r"$[B,\ 12,\ L,\ 64]$",
        fill="blue",
    )

    for node in qkv:
        arrow(ax, node["bottom"], split["top"])

    score = box(
        ax, 1.0, 6.65, 8.0, 1.2,
        r"$A=\mathrm{softmax}(QK^{T}/\sqrt{64}+M)$",
        r"Per-head weights: $[B,\ 12,\ L,\ L]$",
        fill="purple",
    )
    connect(ax, split, score)

    aggregate = box(
        ax, 1.0, 4.65, 8.0, 1.1,
        r"Weighted aggregation: $AV$",
        r"$[B,\ 12,\ L,\ 64]$",
        fill="purple",
    )
    connect(ax, score, aggregate)

    # V 从拆分后的张量直接参与 AV，不经过 QK 分数计算。
    ax.plot(
        [9.0, 9.65, 9.65],
        [9.25, 9.25, 5.2],
        color=COLORS["line"],
        linewidth=1.05,
    )
    arrow(ax, (9.65, 5.2), aggregate["right"])
    ax.text(
        9.5, 7.2, "V",
        ha="center", va="center",
        fontsize=9,
        color=COLORS["muted"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )

    concat = box(
        ax, 1.0, 2.65, 8.0, 1.1,
        "Concatenate head outputs",
        r"$[B,\ L,\ 12 \times 64] = [B,\ L,\ 768]$",
        fill="blue",
    )
    connect(ax, aggregate, concat)

    projection = box(
        ax, 1.0, 0.65, 8.0, 1.1,
        "Output projection",
        r"Linear $768 \rightarrow 768$",
        fill="purple",
    )
    connect(ax, concat, projection)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 10),
        gridspec_kw={"wspace": 0.12},
    )
    fig.patch.set_facecolor("white")

    draw_overview(axes[0])
    draw_encoder(axes[1])
    draw_attention(axes[2])

    fig.suptitle(
        "BERT for Target-Company Sentiment Classification",
        fontsize=17,
        fontweight="bold",
        color=COLORS["ink"],
        y=0.97,
    )

    fig.text(
        0.5, 0.923,
        "12 layers  |  hidden size 768  |  12 attention heads  |  "
        "head dimension 64  |  FFN width 3072",
        ha="center",
        fontsize=10,
        color=COLORS["muted"],
    )

    fig.text(
        0.5, 0.048,
        "B: batch size; L: sequence length (≤ 512); "
        "M: additive padding mask. Dropout is omitted in evaluation mode.",
        ha="center",
        fontsize=9,
        color=COLORS["muted"],
    )
    fig.text(
        0.5, 0.027,
        "Attention matrices illustrate the mathematics; "
        "an optimized SDPA implementation need not explicitly materialize them.",
        ha="center",
        fontsize=8.5,
        color=COLORS["muted"],
    )

    fig.subplots_adjust(
        left=0.025,
        right=0.975,
        top=0.89,
        bottom=0.085,
    )

    for extension in ("png", "pdf", "svg"):
        path = OUTPUT_DIR / f"bert_architecture.{extension}"
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        print("Saved:", path)

    plt.close(fig)


if __name__ == "__main__":
    main()