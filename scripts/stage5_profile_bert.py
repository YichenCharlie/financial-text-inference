from pathlib import Path
from datetime import datetime
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import transformers
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, BertForSequenceClassification


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/stage3_bert_full"
VAL_PATH = ROOT / "data/processed/validation_samples.jsonl"

WARMUP = 10
PROFILE_REPEATS = 3
TOP_K = 15


def device_time_us(event, self_time=False):
    """兼容不同 PyTorch 版本的 CUDA/device 计时字段。"""
    field = "self_device_time_total" if self_time else "device_time_total"
    legacy = "self_cuda_time_total" if self_time else "cuda_time_total"

    if hasattr(event, field):
        return float(getattr(event, field))
    return float(getattr(event, legacy, 0.0))


def add_module_labels(model):
    """给原生 forward 加观察标签，不替换计算。"""
    modules = [
        ("Embedding", model.bert.embeddings),
    ]
    modules.extend(
        (f"Encoder_{i + 1:02d}", layer)
        for i, layer in enumerate(model.bert.encoder.layer)
    )
    modules.extend([
        ("Pooler", model.bert.pooler),
        ("Classifier", model.classifier),
    ])

    handles = []

    def attach(name, module):
        contexts = []

        def before(_module, _inputs):
            context = record_function(f"MODULE::{name}")
            context.__enter__()
            contexts.append(context)

        def after(_module, _inputs, _output):
            contexts.pop().__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))

    for name, module in modules:
        attach(name, module)

    return handles, [name for name, _ in modules]


def save_operator_plot(rows, output_dir):
    selected = rows[:TOP_K]
    total = sum(row["self_gpu_ms_per_forward"] for row in rows)

    labels = [row["name"] for row in selected]
    values = [row["self_gpu_ms_per_forward"] for row in selected]

    remaining = total - sum(values)
    if remaining > 1e-9:
        labels.append("Other operators")
        values.append(remaining)

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(
        labels, values,
        color="#527AA3",
        height=0.65,
    )
    ax.invert_yaxis()

    for bar, value in zip(bars, values):
        ax.text(
            value + total * 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f} ms  ({100 * value / total:.1f}%)",
            va="center", fontsize=9,
        )

    ax.set_xlim(0, max(values) * 1.45)
    ax.set_xlabel("Attributed self GPU time per forward (ms)")
    ax.set_title(
        "Where does GPU computation time go?",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5, 0.015,
        "Operator self time excludes child operators. "
        "Percentages refer to the collected operator self GPU time, not wall latency.",
        ha="center", fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"operator_gpu_time.{suffix}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)


def save_module_plot(rows, output_dir):
    labels = [row["name"] for row in rows]
    values = [row["gpu_ms_per_forward"] for row in rows]
    colors = [
        "#527AA3" if name.startswith("Encoder") else "#74A58A"
        for name in labels
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(labels, values, color=colors, height=0.65)
    ax.invert_yaxis()

    largest = max(values, default=0.0)
    if largest > 0:
        ax.set_xlim(0, largest * 1.3)

    for bar, value in zip(bars, values):
        ax.text(
            value + largest * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center", fontsize=9,
        )

    ax.set_xlabel("Attributed GPU time per forward (ms)")
    ax.set_title(
        "GPU computation attributed to model modules",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5, 0.015,
        "Module totals include child operators. "
        "These labels cover separate modules; do not add them to the operator chart.",
        ha="center", fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"module_gpu_time.{suffix}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5 需要 CUDA GPU，请在 AutoDL GPU 环境运行。")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_dir = ROOT / "results/stage5_profiling" / run_id
    figure_dir = ROOT / "figures/stage5_profiling" / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

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
    inputs = {name: value.cuda() for name, value in inputs.items()}

    metadata = {
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "dtype": str(next(model.parameters()).dtype),
        "input_shape": list(inputs["input_ids"].shape),
        "effective_tokens": inputs["attention_mask"].sum(dim=1).tolist(),
        "attention_class": type(
            model.bert.encoder.layer[0].attention.self
        ).__name__,
        "article_id": sample["article_id"],
        "company": sample["company"],
        "warmup": WARMUP,
        "profile_repeats": PROFILE_REPEATS,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }

    print("=== 实验配置 ===")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    # 预热不带标签、不录制；输入准备和传输已完成。
    print("\n正在预热...")
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(**inputs)
    torch.cuda.synchronize()

    handles, module_names = add_module_labels(model)

    print("正在记录 CPU / GPU 活动...")
    try:
        with torch.inference_mode():
            with profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                for i in range(PROFILE_REPEATS):
                    with record_function(f"FORWARD_{i + 1:02d}"):
                        output = model(**inputs)

                # 等待已提交的 GPU 工作结束。
                # 不在每层同步，以免人为改变执行节奏。
                torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()

    trace_path = result_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))

    averages = list(prof.key_averages())
    by_name = {event.key: event for event in averages}

    # 只统计 CPU 侧算子关联的 GPU self time。
    # 不再把独立 CUDA kernel 事件加一遍，避免重复计数。
    operators = []
    for event in averages:
        if event.device_type != torch.autograd.DeviceType.CPU:
            continue
        if not event.key.startswith("aten::"):
            continue

        self_gpu_us = device_time_us(event, self_time=True)
        if self_gpu_us <= 0:
            continue

        operators.append({
            "name": event.key,
            "calls_per_forward": event.count / PROFILE_REPEATS,
            "self_gpu_ms_per_forward": (
                self_gpu_us / 1000 / PROFILE_REPEATS
            ),
        })

    operators.sort(
        key=lambda row: row["self_gpu_ms_per_forward"],
        reverse=True,
    )

    modules = []
    for name in module_names:
        event = by_name.get(f"MODULE::{name}")
        if event is None:
            raise RuntimeError(f"没有采集到模块标签：{name}")

        modules.append({
            "name": name,
            "gpu_ms_per_forward": (
                device_time_us(event) / 1000 / PROFILE_REPEATS
            ),
        })

    if not operators:
        raise RuntimeError(
            f"没有采集到有效 CUDA 算子时间。已保存 trace：{trace_path}\n"
            "请把控制台中的 profiler/CUPTI 警告贴给我，"
            "不要把空结果解释成 GPU 没有开销。"
        )

    total_self_ms = sum(
        row["self_gpu_ms_per_forward"] for row in operators
    )

    metadata["prediction"] = output.logits.argmax(dim=-1).item()
    metadata["probabilities"] = (
        output.logits.softmax(dim=-1)[0].cpu().tolist()
    )

    summary = {
        "metadata": metadata,
        "operator_self_gpu_ms_per_forward": total_self_ms,
        "operators": operators,
        "modules": modules,
        "timing_note": (
            "Profiler-attributed GPU execution times, averaged over "
            "recorded forwards. Not unprofiled request latency."
        ),
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== GPU 算子耗时 Top 15 ===")
    print(f"{'Operator':<52} {'ms/forward':>12} {'Share':>9} {'Calls/fwd':>10}")
    for row in operators[:TOP_K]:
        value = row["self_gpu_ms_per_forward"]
        print(
            f"{row['name']:<52} "
            f"{value:>12.4f} "
            f"{100 * value / total_self_ms:>8.1f}% "
            f"{row['calls_per_forward']:>10.1f}"
        )

    print("\n=== 模块关联的 GPU 时间 ===")
    for row in modules:
        print(f"{row['name']:<18} {row['gpu_ms_per_forward']:.4f} ms")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "font.size": 10,
    })
    save_operator_plot(operators, figure_dir)
    save_module_plot(modules, figure_dir)

    print("\n=== 输出位置 ===")
    print("结果与时间线:", result_dir)
    print("可视化图表:", figure_dir)
    print("Prediction:", metadata["prediction"])
    print("Probabilities:", metadata["probabilities"])
    print("\n注意：本次 profiler 时间用于分析开销分布，不替代正式 benchmark。")


if __name__ == "__main__":
    main()