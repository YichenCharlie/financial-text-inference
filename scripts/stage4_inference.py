from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, BertForSequenceClassification


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/stage3_bert_full"
VAL_PATH = ROOT / "data/processed/validation_samples.jsonl"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def predict(samples, tokenizer, model):
    companies = [item["company"] for item in samples]
    news = [
        item["title"] + "\n" + item["text"]
        for item in samples
    ]

    # 同一个 batch 的样本补齐到该 batch 的最长输入。
    inputs = tokenizer(
        companies,
        news,
        truncation="only_second",
        max_length=512,
        padding=True,
        return_tensors="pt",
    )

    print("\nBatch size:", len(samples))

    for name, tensor in inputs.items():
        print(
            f"{name}: shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, device={tensor.device}"
        )

    print(
        "每条样本的有效 token 数:",
        inputs["attention_mask"].sum(dim=1).tolist(),
    )

    # Tokenizer 先生成 CPU tensors，再搬到模型所在设备。
    inputs = {
        name: tensor.to(DEVICE)
        for name, tensor in inputs.items()
    }

    # 推理时不构建用于反向传播的计算图。
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

    print("模型是否处于 training 模式:", model.training)
    print("Logits shape:", tuple(logits.shape))
    print("Logits device:", logits.device)
    print("Logits requires_grad:", logits.requires_grad)

    for i, sample in enumerate(samples):
        print(f"\nSample {i + 1}")
        print("Company:", sample["company"])
        print("Label:", sample["label"])
        print("Logits:", logits[i].cpu().tolist())
        print(
            "Probability [non-negative, negative]:",
            probabilities[i].cpu().tolist(),
        )
        print("Prediction:", predictions[i].item())

    return probabilities.cpu()


def main():
    samples = []

    with VAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) == 4:
                break

    if len(samples) < 4:
        raise ValueError("Validation 文件中不足 4 条样本。")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    ).to(DEVICE)

    model.eval()

    print("Device:", DEVICE)
    print("模型参数 dtype:", next(model.parameters()).dtype)

    print("\n=== 单条 inference ===")
    single_probs = predict(samples[:1], tokenizer, model)

    print("\n=== Batch inference ===")
    batch_probs = predict(samples, tokenizer, model)

    # 同一条 sample，单独预测与放进 batch 后进行对照。
    difference = (
        single_probs[0] - batch_probs[0]
    ).abs().max().item()

    print("\n第一条 sample：")
    print("单条预测与 batch 预测的最大概率差:", difference)
    print(
        "预测类别是否一致:",
        single_probs[0].argmax().item()
        == batch_probs[0].argmax().item(),
    )
    benchmark_forward(samples[:1], tokenizer, model)
    benchmark_forward(samples, tokenizer, model)

def benchmark_forward(samples, tokenizer, model, warmup=10, repeats=100):
    companies = [item["company"] for item in samples]
    news = [
        item["title"] + "\n" + item["text"]
        for item in samples
    ]

    # 输入准备和 CPU → GPU 传输放在计时区间外。
    inputs = tokenizer(
        companies,
        news,
        truncation="only_second",
        max_length=512,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        name: tensor.to(DEVICE)
        for name, tensor in inputs.items()
    }

    if DEVICE.type != "cuda":
        raise RuntimeError("这一步使用 CUDA Events，需要在 GPU 上运行。")

    model.eval()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        # 先预热，减少初次执行带来的影响。
        for _ in range(warmup):
            model(**inputs)

        torch.cuda.synchronize()

        start.record()

        for _ in range(repeats):
            model(**inputs)

        end.record()
        end.synchronize()

    total_ms = start.elapsed_time(end)
    mean_batch_ms = total_ms / repeats
    samples_per_second = len(samples) * repeats / (total_ms / 1000)

    print("\n=== Forward benchmark ===")
    print("Input shape:", tuple(inputs["input_ids"].shape))
    print("Warmup:", warmup)
    print("Repeats:", repeats)
    print(f"平均每个 batch 的 forward 耗时: {mean_batch_ms:.3f} ms")
    print(f"Forward throughput: {samples_per_second:.2f} samples/s")


if __name__ == "__main__":
    main()
