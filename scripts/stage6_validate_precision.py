from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, BertForSequenceClassification


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/stage3_bert_full"
VAL_PATH = ROOT / "data/processed/validation_samples.jsonl"
RESULT_DIR = ROOT / "results/stage6_precision"

BATCH_SIZE = 16
EXPECTED_SAMPLES = 1942


def classification_metrics(labels, predictions):
    # 行是真实标签，列是预测标签，顺序均为 [0, 1]。
    matrix = [[0, 0], [0, 0]]

    for label, prediction in zip(labels, predictions):
        matrix[label][prediction] += 1

    per_class = {}
    f1_scores = []

    for cls in (0, 1):
        tp = matrix[cls][cls]
        fp = sum(matrix[row][cls] for row in (0, 1) if row != cls)
        fn = sum(matrix[cls][col] for col in (0, 1) if col != cls)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        per_class[str(cls)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[cls]),
        }
        f1_scores.append(f1)

    return {
        "accuracy": (matrix[0][0] + matrix[1][1]) / len(labels),
        "macro_f1": sum(f1_scores) / 2,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def infer(model, inputs, use_amp):
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=use_amp,
    ):
        logits = model(**inputs).logits

    # 统一在 FP32 中计算 softmax 和输出差异。
    logits = logits.float()

    if not torch.isfinite(logits).all().item():
        raise RuntimeError(
            f"{'AMP FP16' if use_amp else 'FP32'} logits 存在 NaN 或 Inf。"
        )

    probabilities = torch.softmax(logits, dim=-1)

    if not torch.isfinite(probabilities).all().item():
        raise RuntimeError("概率中存在 NaN 或 Inf。")

    return logits.cpu(), probabilities.cpu()


def print_metrics(name, metrics):
    print(f"\n=== {name} ===")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")
    print("Confusion matrix，行=true，列=prediction，顺序=[0, 1]:")
    for row in metrics["confusion_matrix"]:
        print(row)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("请在 CUDA GPU 环境运行。")

    with VAL_PATH.open(encoding="utf-8") as f:
        samples = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(samples) != EXPECTED_SAMPLES:
        raise ValueError(
            f"预期 {EXPECTED_SAMPLES} 条验证样本，实际为 {len(samples)}。"
            "请检查是否使用了同一份验证数据。"
        )

    if any(item["label"] not in (0, 1) for item in samples):
        raise ValueError("验证集包含非 0/1 标签。")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True
    ).to(device="cuda", dtype=torch.float32)
    model.eval()

    print("GPU:", torch.cuda.get_device_name())
    print("Validation samples:", len(samples))
    print("Batch size:", BATCH_SIZE)
    print("Parameter dtype:", next(model.parameters()).dtype)
    print("TF32 enabled:", torch.backends.cuda.matmul.allow_tf32)
    print("本脚本检查分类质量，不测推理速度。")

    fp32_logits_parts = []
    amp_logits_parts = []
    fp32_probs_parts = []
    amp_probs_parts = []

    with torch.inference_mode():
        for start in range(0, len(samples), BATCH_SIZE):
            batch = samples[start:start + BATCH_SIZE]

            inputs = tokenizer(
                [item["company"] for item in batch],
                [
                    item["title"] + "\n" + item["text"]
                    for item in batch
                ],
                truncation="only_second",
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            inputs = {
                name: tensor.to("cuda")
                for name, tensor in inputs.items()
            }

            # 两种模式复用同一批 inputs。
            fp32_logits, fp32_probs = infer(model, inputs, use_amp=False)
            amp_logits, amp_probs = infer(model, inputs, use_amp=True)

            fp32_logits_parts.append(fp32_logits)
            amp_logits_parts.append(amp_logits)
            fp32_probs_parts.append(fp32_probs)
            amp_probs_parts.append(amp_probs)

            completed = start + len(batch)
            if completed % (BATCH_SIZE * 20) == 0 or completed == len(samples):
                print(f"Processed: {completed}/{len(samples)}")

    fp32_logits = torch.cat(fp32_logits_parts)
    amp_logits = torch.cat(amp_logits_parts)
    fp32_probs = torch.cat(fp32_probs_parts)
    amp_probs = torch.cat(amp_probs_parts)

    labels = torch.tensor([item["label"] for item in samples])
    fp32_preds = fp32_logits.argmax(dim=-1)
    amp_preds = amp_logits.argmax(dim=-1)

    fp32_metrics = classification_metrics(
        labels.tolist(), fp32_preds.tolist()
    )
    amp_metrics = classification_metrics(
        labels.tolist(), amp_preds.tolist()
    )

    changed = fp32_preds != amp_preds
    fp32_correct = fp32_preds == labels
    amp_correct = amp_preds == labels

    correct_to_wrong = int((fp32_correct & ~amp_correct).sum().item())
    wrong_to_correct = int((~fp32_correct & amp_correct).sum().item())

    logits_error = (fp32_logits - amp_logits).abs()
    # 每条样本先取两个类别概率差的最大值，再汇总。
    sample_prob_error = (fp32_probs - amp_probs).abs().amax(dim=-1)

    comparison = {
        "changed_predictions": int(changed.sum().item()),
        "prediction_agreement": float((~changed).float().mean().item()),
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_correct": wrong_to_correct,
        "accuracy_delta": amp_metrics["accuracy"] - fp32_metrics["accuracy"],
        "macro_f1_delta": amp_metrics["macro_f1"] - fp32_metrics["macro_f1"],
        "max_absolute_logits_error": logits_error.max().item(),
        "mean_sample_max_probability_error": sample_prob_error.mean().item(),
        "p95_sample_max_probability_error": torch.quantile(
            sample_prob_error, 0.95
        ).item(),
        "max_probability_error": sample_prob_error.max().item(),
        "all_outputs_finite": True,
    }

    print_metrics("FP32", fp32_metrics)
    print_metrics("AMP FP16", amp_metrics)

    print("\n=== 精度对照 ===")
    print("预测不一致数量:", comparison["changed_predictions"])
    print(f"预测一致率: {comparison['prediction_agreement']:.4%}")
    print("FP32 正确 → AMP 错误:", correct_to_wrong)
    print("FP32 错误 → AMP 正确:", wrong_to_correct)
    print(
        "Accuracy 变化:",
        f"{comparison['accuracy_delta'] * 100:+.4f} 个百分点",
    )
    print(
        "Macro-F1 变化:",
        f"{comparison['macro_f1_delta']:+.6f}",
    )
    print("最大 logits 绝对差异:", comparison["max_absolute_logits_error"])
    print(
        "样本最大概率差的均值:",
        comparison["mean_sample_max_probability_error"],
    )
    print(
        "样本最大概率差的 P95:",
        comparison["p95_sample_max_probability_error"],
    )
    print("全部样本最大概率差:", comparison["max_probability_error"])
    print("所有输出均为有限值:", comparison["all_outputs_finite"])

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "samples": len(samples),
        "batch_size": BATCH_SIZE,
        "gpu": torch.cuda.get_device_name(),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "fp32": fp32_metrics,
        "amp_fp16": amp_metrics,
        "comparison": comparison,
    }

    summary_path = RESULT_DIR / "validation_summary.json"
    predictions_path = RESULT_DIR / "validation_comparison.jsonl"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with predictions_path.open("w", encoding="utf-8") as f:
        for index, sample in enumerate(samples):
            record = {
                "article_id": sample["article_id"],
                "company": sample["company"],
                "title": sample["title"],
                "label": sample["label"],
                "fp32_prediction": fp32_preds[index].item(),
                "amp_prediction": amp_preds[index].item(),
                "fp32_probability": fp32_probs[index].tolist(),
                "amp_probability": amp_probs[index].tolist(),
                "prediction_changed": changed[index].item(),
                "max_probability_error": sample_prob_error[index].item(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nSaved:", summary_path)
    print("Saved:", predictions_path)


if __name__ == "__main__":
    main()