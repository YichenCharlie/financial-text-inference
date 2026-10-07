from pathlib import Path
from collections import defaultdict
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, BertForSequenceClassification


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/stage3_bert_full"
VAL_PATH = ROOT / "data/processed/validation_samples.jsonl"

RESULT_DIR = ROOT / "results/stage6_precision/profiling"
FIGURE_DIR = ROOT / "figures/stage6_precision"

WARMUP = 20
REPEATS = 5

GROUPS = [
    "Linear",
    "Attention core",
    "GELU",
    "LayerNorm",
    "Elementwise add",
    "Copy / conversion",
    "Embedding lookup",
    "Other",
]


def forward_once(model, inputs, mode):
    # 与之前的测速一致：每次 forward 单独进入 autocast。
    if mode == "amp_fp16":
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            return model(**inputs).logits

    return model(**inputs).logits


def collect_trace(model, inputs, mode):
    with torch.inference_mode():
        for _ in range(WARMUP):
            forward_once(model, inputs, mode)
        torch.cuda.synchronize()

        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for index in range(REPEATS):
                with record_function(f"{mode}_forward_{index + 1}"):
                    forward_once(model, inputs, mode)

            torch.cuda.synchronize()

    path = RESULT_DIR / f"{mode}_trace.json"
    prof.export_chrome_trace(str(path))
    return path


def classify_operator(name):
    if name in ("aten::addmm", "aten::mm", "aten::linear"):
        return "Linear"

    # bmm / softmax 在当前 BERT forward 中通常属于 attention 核心。
    if (
        "attention" in name.lower()
        or name in (
            "aten::bmm",
            "aten::baddbmm",
            "aten::_softmax",
            "aten::softmax",
        )
    ):
        return "Attention core"

    if "gelu" in name:
        return "GELU"
    if "layer_norm" in name:
        return "LayerNorm"
    if name in ("aten::add", "aten::add_"):
        return "Elementwise add"
    if name in ("aten::copy_", "aten::_to_copy", "aten::to"):
        return "Copy / conversion"
    if name in ("aten::index_select", "aten::embedding"):
        return "Embedding lookup"

    return "Other"


def analyze_trace(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data["traceEvents"]

    # GPU kernel 的 External id 指向对应的 CPU 算子。
    ops = {
        event["args"]["External id"]: event
        for event in events
        if event.get("cat") == "cpu_op"
        and "External id" in event.get("args", {})
    }

    # 在 kernel 没有直接关联信息时，尝试通过 CUDA correlation 查找。
    runtime_ids = {
        event["args"]["correlation"]: event["args"].get("External id")
        for event in events
        if event.get("cat") == "cuda_runtime"
        and "correlation" in event.get("args", {})
    }

    totals = {group: 0.0 for group in GROUPS}
    operators = defaultdict(lambda: {"duration_us": 0.0, "kernels": 0})
    kernel_details = defaultdict(
        lambda: {"duration_us": 0.0, "count": 0}
    )
    unmatched = 0

    kernels = [
        event for event in events
        if event.get("cat") == "kernel"
        and event.get("ph") == "X"
    ]
    if not kernels:
        raise RuntimeError(f"{path} 中没有 GPU kernel 记录。")

    for kernel in kernels:
        args = kernel.get("args", {})
        op = ops.get(args.get("External id"))

        if op is None:
            linked_id = runtime_ids.get(args.get("correlation"))
            op = ops.get(linked_id)

        op_name = op["name"] if op else "Unattributed"
        if op is None:
            unmatched += 1

        group = classify_operator(op_name)
        duration = float(kernel["dur"])

        totals[group] += duration / 1000 / REPEATS
        operators[op_name]["duration_us"] += duration
        operators[op_name]["kernels"] += 1

        key = (group, op_name, kernel["name"])
        kernel_details[key]["duration_us"] += duration
        kernel_details[key]["count"] += 1

    operator_rows = sorted(
        [
            {
                "operator": name,
                "kernel_ms_per_forward": values["duration_us"] / 1000 / REPEATS,
                "kernels_per_forward": values["kernels"] / REPEATS,
            }
            for name, values in operators.items()
        ],
        key=lambda row: row["kernel_ms_per_forward"],
        reverse=True,
    )

    kernel_rows = sorted(
        [
            {
                "group": group,
                "operator": op_name,
                "kernel_name": kernel_name,
                "kernel_ms_per_forward": values["duration_us"] / 1000 / REPEATS,
                "kernels_per_forward": values["count"] / REPEATS,
            }
            for (group, op_name, kernel_name), values in kernel_details.items()
        ],
        key=lambda row: row["kernel_ms_per_forward"],
        reverse=True,
    )

    # memcpy/memset 与计算 kernels 分开保存，不混入计算类别图。
    memory_activity = {}
    for category in ("gpu_memcpy", "gpu_memset"):
        matching = [
            event for event in events
            if event.get("cat") == category
            and event.get("ph") == "X"
        ]
        memory_activity[category] = {
            "events_per_forward": len(matching) / REPEATS,
            "ms_per_forward": (
                sum(event.get("dur", 0) for event in matching)
                / 1000 / REPEATS
            ),
        }

    return {
        "group_kernel_ms_per_forward": totals,
        "operators": operator_rows,
        "kernels": kernel_rows,
        "unattributed_kernel_count": unmatched,
        "memory_activity": memory_activity,
    }


def plot_comparison(results):
    fp32 = results["fp32"]["group_kernel_ms_per_forward"]
    amp = results["amp_fp16"]["group_kernel_ms_per_forward"]

    fig, ax = plt.subplots(figsize=(12, 7))
    positions = list(range(len(GROUPS)))

    for offset, values, label, color in [
        (-0.18, fp32, "FP32", "#527AA3"),
        (0.18, amp, "AMP FP16", "#75A48B"),
    ]:
        ax.barh(
            [position + offset for position in positions],
            [values[group] for group in GROUPS],
            height=0.32,
            label=label,
            color=color,
        )
        for position, group in zip(positions, GROUPS):
            value = values[group]
            ax.text(
                value,
                position + offset,
                f"  {value:.3f}",
                va="center",
                fontsize=9,
            )

    largest = max(list(fp32.values()) + list(amp.values()))
    ax.set_xlim(0, largest * 1.2)
    ax.set_yticks(positions, GROUPS)
    ax.invert_yaxis()
    ax.set_xlabel("Sum of GPU kernel durations per forward (ms)")
    ax.set_title(
        "FP32 vs mixed precision: where did GPU work change?",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5, 0.025,
        "Profiled kernel durations, not unprofiled forward latency. "
        "Copy / conversion is not exclusively dtype conversion.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURE_DIR / f"precision_kernel_comparison.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("请在 CUDA GPU 环境运行。")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

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
    print("模型参数 dtype:", next(model.parameters()).dtype)
    print("TF32 enabled:", torch.backends.cuda.matmul.allow_tf32)

    results = {}

    for mode in ("fp32", "amp_fp16"):
        print(f"\n正在记录 {mode} ...")
        trace_path = collect_trace(model, inputs, mode)
        results[mode] = analyze_trace(trace_path)

        print("Trace:", trace_path)
        print(
            "无法关联 CPU 算子的 kernel 数:",
            results[mode]["unattributed_kernel_count"],
        )

        print("Attention kernel 名称：")
        attention_names = {
            row["kernel_name"]
            for row in results[mode]["kernels"]
            if row["group"] == "Attention core"
        }
        for name in sorted(attention_names):
            print(" ", name)

    print("\n=== GPU kernel 时间对照，ms / forward ===")
    print(f"{'Category':<22} {'FP32':>10} {'AMP FP16':>12} {'Change':>12}")

    for group in GROUPS:
        base = results["fp32"]["group_kernel_ms_per_forward"][group]
        mixed = results["amp_fp16"]["group_kernel_ms_per_forward"][group]

        print(
            f"{group:<22} {base:>10.4f} {mixed:>12.4f} "
            f"{mixed - base:>+12.4f}"
        )

    for mode in results:
        print(f"\n=== {mode} 算子对应的 kernel 时间 Top 12 ===")
        for row in results[mode]["operators"][:12]:
            print(
                f"{row['operator']:<48} "
                f"{row['kernel_ms_per_forward']:.4f} ms"
            )

    summary = {
        "gpu": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "input_shape": list(inputs["input_ids"].shape),
        "profile_repeats": REPEATS,
        "autocast_scope": "one context per forward",
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "results": results,
        "note": (
            "Sums of kernel durations grouped by associated CPU operator. "
            "Excludes memcpy/memset from the chart. "
            "Not a replacement for unprofiled benchmarking."
        ),
    }
    (RESULT_DIR / "kernel_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "pdf.fonttype": 42,
    })
    plot_comparison(results)

    print("\n结果:", RESULT_DIR / "kernel_comparison.json")
    print("图表:", FIGURE_DIR / "precision_kernel_comparison.png")


if __name__ == "__main__":
    main()