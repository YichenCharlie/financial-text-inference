from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, BertForSequenceClassification


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/stage3_bert_full"
VAL_PATH = ROOT / "data/processed/validation_samples.jsonl"


def dtype_name(tensor):
    return str(tensor.dtype).replace("torch.", "")


def run_and_observe(model, inputs, use_amp):
    layer = model.bert.encoder.layer[0]

    # 只观察第一层的关键位置和最终分类层。
    targets = [
        ("Embedding", model.bert.embeddings),
        ("Layer1.Q", layer.attention.self.query),
        ("Layer1.K", layer.attention.self.key),
        ("Layer1.V", layer.attention.self.value),
        ("Layer1.O", layer.attention.output.dense),
        ("Layer1.LN1", layer.attention.output.LayerNorm),
        ("Layer1.FFN-up", layer.intermediate.dense),
        ("Layer1.GELU", layer.intermediate.intermediate_act_fn),
        ("Layer1.FFN-down", layer.output.dense),
        ("Layer1.LN2", layer.output.LayerNorm),
        ("Classifier", model.classifier),
    ]

    observations = []
    handles = []

    def make_hook(name):
        def hook(module, args, output):
            input_tensor = next(
                (value for value in args if isinstance(value, torch.Tensor)),
                None,
            )
            parameter = next(module.parameters(recurse=False), None)

            observations.append({
                "name": name,
                "input": (
                    dtype_name(input_tensor)
                    if input_tensor is not None else "-"
                ),
                "parameter": (
                    dtype_name(parameter)
                    if parameter is not None else "-"
                ),
                "output": dtype_name(output),
            })

        return hook

    for name, module in targets:
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(**inputs).logits
    finally:
        for handle in handles:
            handle.remove()

    # 在 autocast 区间外统一转 FP32，再计算概率和比较误差。
    # 这不会恢复之前因低精度计算而丢失的信息。
    logits_fp32 = logits.float()

    if not torch.isfinite(logits_fp32).all().item():
        raise RuntimeError("输出存在 NaN 或 Inf，请先检查数值问题。")

    probabilities = torch.softmax(logits_fp32, dim=-1)

    return {
        "observations": observations,
        "original_logits_dtype": dtype_name(logits),
        "logits": logits_fp32.cpu(),
        "probabilities": probabilities.cpu(),
    }


def print_result(title, result):
    print(f"\n=== {title} ===")
    print(
        f"{'Module':<20}"
        f"{'Input dtype':<15}"
        f"{'Parameter':<15}"
        f"{'Output dtype'}"
    )

    for row in result["observations"]:
        print(
            f"{row['name']:<20}"
            f"{row['input']:<15}"
            f"{row['parameter']:<15}"
            f"{row['output']}"
        )

    print("原始 logits dtype:", result["original_logits_dtype"])
    print("Logits:", result["logits"][0].tolist())
    print(
        "Probability [non-negative, negative]:",
        result["probabilities"][0].tolist(),
    )
    print("Prediction:", result["logits"].argmax(dim=-1).item())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("请在 AutoDL 的 CUDA GPU 环境运行。")

    with VAL_PATH.open(encoding="utf-8") as f:
        sample = next(
            json.loads(line)
            for line in f
            if line.strip()
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

    print("GPU:", torch.cuda.get_device_name())
    print("目标公司:", sample["company"])
    print("真实 label:", sample["label"])
    print("Input shape:", tuple(inputs["input_ids"].shape))
    print("Input IDs dtype:", inputs["input_ids"].dtype)
    print("运行前模型参数 dtype:", next(model.parameters()).dtype)

    fp32 = run_and_observe(model, inputs, use_amp=False)
    mixed = run_and_observe(model, inputs, use_amp=True)

    print_result("FP32 baseline", fp32)
    print_result("FP16 mixed precision", mixed)

    logits_difference = (fp32["logits"] - mixed["logits"]).abs()
    probability_difference = (
        fp32["probabilities"] - mixed["probabilities"]
    ).abs()

    print("\n=== 两种模式的输出对照 ===")
    print("最大 logits 绝对差异:", logits_difference.max().item())
    print("最大概率绝对差异:", probability_difference.max().item())
    print(
        "预测类别是否一致:",
        torch.equal(
            fp32["logits"].argmax(dim=-1),
            mixed["logits"].argmax(dim=-1),
        ),
    )
    print("运行后模型参数 dtype:", next(model.parameters()).dtype)

    print(
        "\n说明：Input 列显示模块收到的 Tensor，"
        "不一定是内部算子经 autocast 转换后的输入。"
    )
    print(
        "Embedding 使用关键字参数调用，因此该行 Input 显示 '-'；"
        "其参数位于子模块中，Parameter 也显示 '-'。"
    )


if __name__ == "__main__":
    main()