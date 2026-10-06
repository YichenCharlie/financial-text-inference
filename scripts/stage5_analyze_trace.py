from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, ConnectionPatch


ROOT = Path(__file__).resolve().parent.parent

COLORS = {
    "Linear": "#527AA3",
    "Attention": "#9474B4",
    "GELU": "#D79A4C",
    "LayerNorm": "#66A389",
    "Add": "#C67979",
    "Other": "#A5ACB5",
}


def end(event):
    return event["ts"] + event.get("dur", 0)


def inside(event, parent):
    return (
        event.get("pid") == parent.get("pid")
        and event.get("tid") == parent.get("tid")
        and parent["ts"] <= event["ts"] < end(parent)
    )


def external_id(event):
    return event.get("args", {}).get("External id")


def category(name):
    if name in ("aten::linear", "aten::addmm", "aten::mm"):
        return "Linear"
    if "attention" in name:
        return "Attention"
    if "gelu" in name:
        return "GELU"
    if "layer_norm" in name:
        return "LayerNorm"
    if name in ("aten::add", "aten::add_"):
        return "Add"
    return "Other"


def save_figure(fig, directory, name):
    for extension in ("png", "pdf"):
        path = directory / f"{name}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="已有 trace.json 的路径",
    )
    parser.add_argument("--forward", type=int, default=2)
    parser.add_argument("--layer", type=int, default=2)
    args = parser.parse_args()

    trace_path = args.trace.expanduser().resolve()
    data = json.loads(trace_path.read_text(encoding="utf-8"))

    # 只使用带开始时间和持续时间的完整区间事件。
    events = [
        event for event in data["traceEvents"]
        if event.get("ph") == "X"
        and "ts" in event
        and "dur" in event
    ]

    cpu_ops = [
        event for event in events
        if event.get("cat") == "cpu_op"
    ]
    ops_by_id = {
        external_id(event): event
        for event in cpu_ops
        if external_id(event) is not None
    }

    kernels = [
        event for event in events
        if event.get("cat") == "kernel"
    ]
    activities = [
        event for event in events
        if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    modules = [
        event for event in events
        if event.get("cat") == "user_annotation"
        and event["name"].startswith("MODULE::")
    ]

    if not kernels:
        raise RuntimeError("trace 中没有 CUDA kernel 事件。")

    def belongs_to(event, module):
        # 根据 CPU 算子关联信息归属模块。
        # 不用 CPU/GPU 时间区间重叠来猜测归属。
        op = ops_by_id.get(external_id(event))
        return op is not None and inside(op, module)

    output_dir = (
        ROOT / "figures/stage5_profiling"
        / trace_path.parent.name / "trace_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1. 修正模块图：kernel 总时长与首尾区间分别统计。
    # -------------------------------------------------
    grouped = {}
    instance_records = []

    for module in sorted(modules, key=lambda event: event["ts"]):
        matched = [
            kernel for kernel in kernels
            if belongs_to(kernel, module)
        ]
        if not matched:
            continue

        name = module["name"].replace("MODULE::", "")
        kernel_sum_ms = sum(k["dur"] for k in matched) / 1000
        span_ms = (
            max(end(k) for k in matched)
            - min(k["ts"] for k in matched)
        ) / 1000

        record = {
            "module": name,
            "kernel_count": len(matched),
            "kernel_sum_ms": kernel_sum_ms,
            "kernel_span_ms": span_ms,
        }
        grouped.setdefault(name, []).append(record)
        instance_records.append(record)

    if not grouped:
        raise RuntimeError(
            "无法把 kernels 关联到模块，请保留 trace 并贴出错误。"
        )

    names = list(grouped)
    mean_sum = [
        sum(r["kernel_sum_ms"] for r in grouped[name])
        / len(grouped[name])
        for name in names
    ]
    mean_span = [
        sum(r["kernel_span_ms"] for r in grouped[name])
        / len(grouped[name])
        for name in names
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    positions = list(range(len(names)))

    ax.barh(
        [y - 0.18 for y in positions], mean_sum,
        height=0.32, color=COLORS["Linear"],
        label="Sum of kernel execution durations",
    )
    ax.barh(
        [y + 0.18 for y in positions], mean_span,
        height=0.32, color="#C7D4E2",
        label="First-kernel start to last-kernel end",
    )

    for y, value in zip(positions, mean_sum):
        ax.text(
            value, y - 0.18, f"  {value:.3f}",
            va="center", fontsize=8,
        )
    for y, value in zip(positions, mean_span):
        ax.text(
            value, y + 0.18, f"  {value:.3f}",
            va="center", fontsize=8,
        )

    ax.set_yticks(positions, names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(mean_sum + mean_span) * 1.22)
    ax.set_xlabel("Mean time per module invocation (ms)")
    ax.set_title(
        "Module timing: kernel work versus elapsed span",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5, 0.015,
        "Kernel sums exclude memcpy/memset. Spans may include gaps. "
        "For overlapping kernels, sums are not GPU busy-time unions.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, output_dir, "module_kernel_time")

    # -------------------------------------------------
    # 2. 选择一个 forward 中的一个 Encoder layer。
    # -------------------------------------------------
    forward_name = f"FORWARD_{args.forward:02d}"
    layer_name = f"MODULE::Encoder_{args.layer:02d}"

    forwards = [
        event for event in events
        if event.get("cat") == "user_annotation"
        and event["name"] == forward_name
    ]
    if len(forwards) != 1:
        raise RuntimeError(f"未找到唯一的 {forward_name}")

    candidates = [
        module for module in modules
        if module["name"] == layer_name
        and inside(module, forwards[0])
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"未找到唯一的 {layer_name}")

    selected_module = candidates[0]

    selected_names = {
        "aten::linear",
        "aten::_efficient_attention_forward",
        "aten::_scaled_dot_product_flash_attention",
        "aten::gelu",
        "aten::native_layer_norm",
        "aten::add",
    }
    selected_ops = sorted(
        [
            op for op in cpu_ops
            if inside(op, selected_module)
            and op["name"] in selected_names
        ],
        key=lambda event: event["ts"],
    )

    selected_gpu = sorted(
        [
            event for event in activities
            if belongs_to(event, selected_module)
        ],
        key=lambda event: event["ts"],
    )
    selected_kernels = [
        event for event in selected_gpu
        if event["cat"] == "kernel"
    ]
    if not selected_kernels:
        raise RuntimeError("选定层没有关联到 GPU kernels。")

    # 将嵌套的 addmm 等算子归入上层 Linear，用于统一颜色与标签。
    def owner(event):
        op = ops_by_id.get(external_id(event))
        if op is None:
            return None
        containers = [
            parent for parent in selected_ops
            if inside(op, parent)
            and end(op) <= end(parent) + 0.1
        ]
        return min(containers, key=lambda e: e["dur"]) if containers else op

    linears = [
        op for op in selected_ops if op["name"] == "aten::linear"
    ]
    linear_names = ["Q", "K", "V", "O", "FFN-up", "FFN-down"]
    labels = {
        id(op): (
            linear_names[i]
            if len(linears) == 6 else f"Linear-{i + 1}"
        )
        for i, op in enumerate(linears)
    }

    streams = {
        event.get("args", {}).get("stream")
        for event in selected_gpu
    }
    if len(streams) != 1:
        raise RuntimeError(
            f"选定层涉及多个 streams：{streams}。"
            "此脚本按单 stream 展示，不将多个 stream 合并误画。"
        )

    runtime_events = [
        event for event in events
        if event.get("cat") == "cuda_runtime"
        and inside(event, selected_module)
        and (
            "Launch" in event["name"]
            or "Memcpy" in event["name"]
            or "Memset" in event["name"]
        )
    ]

    origin = min(
        selected_module["ts"],
        min(event["ts"] for event in selected_gpu),
    )

    fig, ax = plt.subplots(figsize=(16, 5.8))
    lane_y = {"CPU": 2.0, "CUDA": 1.0, "GPU": 0.0}

    def draw_event(event, lane, color, label=None):
        start = (event["ts"] - origin) / 1000
        duration = event["dur"] / 1000
        ax.broken_barh(
            [(start, duration)],
            (lane_y[lane] - 0.18, 0.36),
            facecolors=color,
            edgecolors="white",
            linewidth=0.3,
        )
        if label:
            ax.text(
                start + duration / 2,
                lane_y[lane] + 0.27,
                label,
                rotation=45,
                ha="left", va="bottom",
                fontsize=8,
            )

    for op in selected_ops:
        draw_event(
            op, "CPU", COLORS[category(op["name"])],
            labels.get(id(op)),
        )

    for event in runtime_events:
        parent = owner(event)
        name = parent["name"] if parent else ""
        draw_event(event, "CUDA", COLORS[category(name)])

    for event in selected_gpu:
        parent = owner(event)
        name = parent["name"] if parent else ""
        draw_event(
            event, "GPU",
            COLORS[category(name)] if event["cat"] == "kernel"
            else COLORS["Other"],
        )

    # 虚线连接每个 Linear 与它的第一个 GPU kernel。
    # 连接表示归属，不表示同步等待。
    for op in linears:
        matches = [
            kernel for kernel in selected_kernels
            if owner(kernel) is op
        ]
        if not matches:
            continue
        kernel = min(matches, key=lambda event: event["ts"])
        connection = ConnectionPatch(
            xyA=((op["ts"] - origin) / 1000, 1.80),
            xyB=((kernel["ts"] - origin) / 1000, 0.20),
            coordsA="data", coordsB="data",
            axesA=ax, axesB=ax,
            color=COLORS["Linear"],
            linewidth=0.7, alpha=0.45,
            linestyle="--", zorder=0,
        )
        ax.add_artist(connection)

    ax.set_yticks(
        [2, 1, 0],
        ["CPU operators\n(selected)", "CUDA submission\n(runtime calls)",
         "GPU execution\n(kernels + memory ops)"],
    )
    ax.set_ylim(-0.6, 3.0)
    ax.set_xlabel("Time relative to selected module / first GPU activity (ms)")
    ax.set_title(
        f"{forward_name} / Encoder {args.layer:02d}: CPU and GPU timeline",
        fontsize=14, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(
        handles=[Patch(color=c, label=n) for n, c in COLORS.items()],
        loc="upper center", ncol=6, fontsize=9,
    )
    fig.text(
        0.5, 0.02,
        "Dashed lines associate Linear CPU operators with their first GPU kernel; "
        "they do not represent synchronization or measured waiting time.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, output_dir, "encoder_cpu_gpu_timeline")

    # 保存可复核的数值与对应关系。
    kernel_sum = sum(k["dur"] for k in selected_kernels) / 1000
    kernel_span = (
        max(end(k) for k in selected_kernels)
        - min(k["ts"] for k in selected_kernels)
    ) / 1000

    report = {
        "source_trace": str(trace_path),
        "selected_forward": forward_name,
        "selected_module": layer_name,
        "cpu_module_ms": selected_module["dur"] / 1000,
        "kernel_sum_ms": kernel_sum,
        "kernel_span_ms": kernel_span,
        "module_invocations": instance_records,
    }
    (output_dir / "timing_details.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== 选定观察对象 ===")
    print(forward_name, layer_name)
    print("CPU 模块区间:", round(selected_module["dur"] / 1000, 4), "ms")
    print("GPU kernel 时间之和:", round(kernel_sum, 4), "ms")
    print("GPU kernel 首尾区间:", round(kernel_span, 4), "ms")
    print("关联 kernel 数:", len(selected_kernels))

    print("\n=== Linear 与 GPU kernels 的对应 ===")
    for op in linears:
        matches = [k for k in selected_kernels if owner(k) is op]
        print(
            f"{labels[id(op)]:<10} "
            f"CPU 区间={op['dur'] / 1000:.4f} ms, "
            f"GPU kernels={len(matches)}, "
            f"kernel 总时长={sum(k['dur'] for k in matches) / 1000:.4f} ms"
        )

    print("\n图片及数值已保存到:", output_dir)
    print("module_kernel_time.png")
    print("encoder_cpu_gpu_timeline.png")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "pdf.fonttype": 42,
    })
    main()