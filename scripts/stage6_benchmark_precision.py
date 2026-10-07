from pathlib import Path
import json
import statistics
import time

import torch
import transformers
from transformers import AutoTokenizer, BertForSequenceClassification


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/stage3_bert_full"
VAL_PATH = ROOT / "data/processed/validation_samples.jsonl"
OUTPUT_PATH = ROOT / "results/stage6_precision/benchmark.json"

WARMUP = 20
REPEATS = 50

ORDERS = [
    ["fp32", "amp_fp16"],
    ["amp_fp16", "fp32"],
    ["fp32", "amp_fp16"],
    ["amp_fp16", "fp32"],
]


def forward_once(model, inputs, mode):
    # 每次调用有独立的 autocast 生命周期。
    # FP32 分支不进入 autocast。
    if mode == "amp_fp16":
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            return model(**inputs).logits

    return model(**inputs).logits


def benchmark(model, inputs, mode):
    # 外层由 main 启用 inference_mode。
    for _ in range(WARMUP):
        forward_once(model, inputs, mode)
    torch.cuda.synchronize()

    # 提前初始化 Events，避免首次初始化进入计时。
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()

    wall_start = time.perf_counter()
    start.record()

    for _ in range(REPEATS):
        forward_once(model, inputs, mode)

    end.record()
    end.synchronize()
    wall_end = time.perf_counter()

    return {
        "cuda_interval_ms": start.elapsed_time(end) / REPEATS,
        "wall_ms": (wall_end - wall_start) * 1000 / REPEATS,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("请在 CUDA GPU 环境运行。")

    with VAL_PATH.open(encoding="utf-8") as f:
        sample = next(
            json.loads(line) for line in f if line.strip()
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    ).to(device="cuda", dtype=torch.float32)
    model.eval()

    inputs = tokenizer(
        sample["company"],
        sample["title"] + "\n" + sample["text"],
        truncation="only_second",
        max_length=512,
        return_tensors="pt",
    )
    inputs = {
        name: tensor.to("cuda")
        for name, tensor in inputs.items()
    }

    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "input_shape": list(inputs["input_ids"].shape),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "company": sample["company"],
        "warmup": WARMUP,
        "repeats_per_block": REPEATS,
        "rounds": len(ORDERS),
        "autocast_scope": "one context per forward",
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }

    print("=== 实验配置 ===")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    records = {"fp32": [], "amp_fp16": []}

    with torch.inference_mode():
        # 在计时之外确认输出正常。
        for mode in records:
            logits = forward_once(model, inputs, mode)
            if not torch.isfinite(logits).all().item():
                raise RuntimeError(f"{mode} 输出存在 NaN 或 Inf。")
            print(
                f"{mode}: logits dtype={logits.dtype}, "
                f"prediction={logits.argmax(dim=-1).item()}"
            )
        torch.cuda.synchronize()

        for round_index, order in enumerate(ORDERS, start=1):
            print(f"\n=== Round {round_index} ===")
            for mode in order:
                result = benchmark(model, inputs, mode)
                records[mode].append(result)
                print(
                    f"{mode:<12} "
                    f"CUDA interval={result['cuda_interval_ms']:.3f} ms, "
                    f"Wall={result['wall_ms']:.3f} ms"
                )

    summary = {
        mode: {
            metric: statistics.median(
                row[metric] for row in results
            )
            for metric in ("cuda_interval_ms", "wall_ms")
        }
        for mode, results in records.items()
    }

    print("\n=== 汇总：4 个计时块均值的中位数 ===")
    for mode, values in summary.items():
        print(
            f"{mode:<12} "
            f"CUDA interval={values['cuda_interval_ms']:.3f} ms, "
            f"Wall={values['wall_ms']:.3f} ms"
        )

    comparisons = {}
    print("\n=== 混合精度相对 FP32 ===")
    for metric in ("cuda_interval_ms", "wall_ms"):
        baseline = summary["fp32"][metric]
        mixed = summary["amp_fp16"][metric]

        speedup = baseline / mixed
        reduction = (1 - mixed / baseline) * 100

        comparisons[metric] = {
            "speedup": speedup,
            "time_reduction_percent": reduction,
        }
        print(
            f"{metric}: 加速比={speedup:.3f}x, "
            f"耗时减少={reduction:.1f}%"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "orders": ORDERS,
                "records": records,
                "summary": summary,
                "comparisons": comparisons,
                "timing_scope": (
                    "Repeated GPU-resident forward calls, including "
                    "per-call autocast overhead; excludes tokenization "
                    "and CPU-to-GPU input transfer."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()