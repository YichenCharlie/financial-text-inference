from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
import json
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer, BertForSequenceClassification

from stage5_profile_bert import add_module_labels


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/stage3_bert_full"
VAL_PATH = ROOT / "data/processed/validation_samples.jsonl"

WARMUP = 20
REPEATS = 20

# 轮换执行顺序，减小固定顺序带来的影响。
ORDERS = [
    ["baseline", "hooks", "profiler"],
    ["hooks", "profiler", "baseline"],
    ["profiler", "baseline", "hooks"],
]

NAMES = {
    "baseline": "Baseline",
    "hooks": "Hooks only",
    "profiler": "Hooks + Profiler",
}


def measure_block(model, inputs):
    # CUDA Event 创建和初始化放在计时区间外。
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    start.record()
    finish.record()
    torch.cuda.synchronize()

    wall_start = time.perf_counter()
    start.record()

    for _ in range(REPEATS):
        model(**inputs)

    finish.record()
    finish.synchronize()
    wall_end = time.perf_counter()

    return {
        "cuda_interval_ms": start.elapsed_time(finish) / REPEATS,
        "wall_ms": (wall_end - wall_start) * 1000 / REPEATS,
    }


def run_condition(condition, model, inputs):
    handles = []

    if condition != "baseline":
        handles, _ = add_module_labels(model)

    try:
        # 每种方式开始前都预热；Profiler 此时还没有开启。
        for _ in range(WARMUP):
            model(**inputs)
        torch.cuda.synchronize()

        context = (
            profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
            )
            if condition == "profiler"
            else nullcontext()
        )

        # 只测记录期间的执行。
        # 不包含 Profiler 启动、退出时处理结果或写文件的时间。
        with context:
            result = measure_block(model, inputs)

        return result

    finally:
        for handle in handles:
            handle.remove()


def plot_results(records, summary, path):
    keys = list(NAMES)
    labels = [NAMES[key] for key in keys]
    metrics = [
        ("cuda_interval_ms", "CUDA Events interval"),
        ("wall_ms", "CPU wall time with GPU completion"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#527AA3", "#75A48B", "#CB9456"]

    for ax, (metric, title) in zip(axes, metrics):
        values = [summary[key][metric] for key in keys]
        bars = ax.bar(labels, values, color=colors, width=0.6)

        for index, key in enumerate(keys):
            points = [row[metric] for row in records[key]]
            ax.scatter(
                [index - 0.08, index, index + 0.08],
                points,
                color="#263445",
                s=22,
                zorder=3,
            )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center", va="bottom", fontsize=10,
            )

        maximum = max(
            row[metric]
            for rows in records.values()
            for row in rows
        )
        ax.set_ylim(0, maximum * 1.2)
        ax.set_title(title)
        ax.set_ylabel("Mean time per forward within a block (ms)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "How much does observation change execution time?",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5, 0.02,
        "Bars: median of 3 block means. Dots: individual block means. "
        "Profiler startup, teardown and file export are excluded.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("请在 CUDA GPU 环境运行。")

    with VAL_PATH.open(encoding="utf-8") as f:
        sample = next(json.loads(line) for line in f if line.strip())

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True
    ).to(device="cuda", dtype=torch.float32)
    model.eval()

    inputs = tokenizer(
        sample["company"],
        sample["title"] + "\n" + sample["text"],
        truncation="only_second",
        max_length=512,
        return_tensors="pt",
    )
    inputs = {name: tensor.cuda() for name, tensor in inputs.items()}

    print("GPU:", torch.cuda.get_device_name())
    print("Input shape:", tuple(inputs["input_ids"].shape))
    print("Dtype:", next(model.parameters()).dtype)
    print("每个计时块的 forward 次数:", REPEATS)

    records = {key: [] for key in NAMES}

    with torch.inference_mode():
        for round_index, order in enumerate(ORDERS, start=1):
            print(f"\n=== Round {round_index} ===")
            for condition in order:
                result = run_condition(condition, model, inputs)
                records[condition].append(result)

                print(
                    f"{NAMES[condition]:<18} "
                    f"CUDA interval={result['cuda_interval_ms']:.3f} ms, "
                    f"Wall={result['wall_ms']:.3f} ms"
                )

    summary = {
        condition: {
            metric: statistics.median(
                row[metric] for row in rows
            )
            for metric in ("cuda_interval_ms", "wall_ms")
        }
        for condition, rows in records.items()
    }

    print("\n=== 汇总：3 个计时块均值的中位数 ===")
    for condition in NAMES:
        result = summary[condition]
        print(
            f"{NAMES[condition]:<18} "
            f"CUDA interval={result['cuda_interval_ms']:.3f} ms, "
            f"Wall={result['wall_ms']:.3f} ms"
        )

    print("\n=== 相对 Baseline 的变化 ===")
    for condition in ("hooks", "profiler"):
        for metric in ("cuda_interval_ms", "wall_ms"):
            change = (
                summary[condition][metric]
                / summary["baseline"][metric] - 1
            ) * 100
            print(f"{NAMES[condition]} / {metric}: {change:+.1f}%")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_dir = ROOT / "results/stage5_overhead" / run_id
    figure_dir = ROOT / "figures/stage5_overhead" / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "gpu": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "input_shape": list(inputs["input_ids"].shape),
        "dtype": str(next(model.parameters()).dtype),
        "warmup": WARMUP,
        "repeats": REPEATS,
        "orders": ORDERS,
        "records": records,
        "summary": summary,
        "note": (
            "Repeated forward blocks with GPU-resident input. "
            "Not end-to-end request latency. Profiler setup, "
            "teardown and export are excluded."
        ),
    }

    (result_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_results(
        records, summary,
        figure_dir / "instrumentation_overhead.png",
    )

    print("\n结果:", result_dir)
    print("图表:", figure_dir / "instrumentation_overhead.png")


if __name__ == "__main__":
    main()