from pathlib import Path
import json
import math

import torch
from transformers import AutoTokenizer, BertForSequenceClassification


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/stage3_bert_full"
VAL_PATH = ROOT / "data/processed/validation_samples.jsonl"


def manual_layer_norm(tensor, module):
    """按每个 token 的最后一维，手动计算 LayerNorm。"""
    mean = tensor.mean(dim=-1, keepdim=True)
    variance = tensor.var(
        dim=-1,
        keepdim=True,
        unbiased=False,
    )

    normalized = (
        (tensor - mean)
        / torch.sqrt(variance + module.eps)
    )

    return normalized * module.weight + module.bias


def split_heads(tensor, num_heads):
    batch_size, seq_length, hidden_size = tensor.shape
    head_dim = hidden_size // num_heads

    return tensor.reshape(
        batch_size,
        seq_length,
        num_heads,
        head_dim,
    ).transpose(1, 2)


def first_values(tensor, position, count=8):
    return [
        round(value, 4)
        for value in tensor[0, position, :count].tolist()
    ]


def compare_outputs(manual, reference):
    difference = (manual - reference).abs()

    print("最大绝对差异:", difference.max().item())
    print("平均绝对差异:", difference.mean().item())
    print(
        "是否在容差内一致:",
        torch.allclose(
            manual,
            reference,
            atol=1e-5,
            rtol=1e-5,
        ),
    )


def main():
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

    # 在 CPU 上验证计算过程，本脚本不测推理性能。
    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model.eval()

    inputs = tokenizer(
        sample["company"],
        sample["title"] + "\n" + sample["text"],
        truncation="only_second",
        max_length=512,
        return_tensors="pt",
    )

    layer = model.bert.encoder.layer[0]
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    # 位置 0 是 [CLS]，位置 1 是目标公司的第一个 token。
    position = 1

    with torch.inference_mode():
        # =============================================
        # 1. Embedding
        # =============================================
        x = model.bert.embeddings(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
        )

        # =============================================
        # 2. Q、K、V 投影与多头注意力
        # =============================================
        attention = layer.attention.self

        q = split_heads(attention.query(x), num_heads)
        k = split_heads(attention.key(x), num_heads)
        v = split_heads(attention.value(x), num_heads)

        # 有效位置为 0，padding 位置为极小负数。
        # 手动路径与原生路径使用相同的加法 mask。
        padding_positions = (
            inputs["attention_mask"][:, None, None, :] == 0
        )

        attention_mask = torch.zeros(
            padding_positions.shape,
            dtype=x.dtype,
            device=x.device,
        ).masked_fill(
            padding_positions,
            torch.finfo(x.dtype).min,
        )

        scores = (
            q @ k.transpose(-1, -2)
        ) / math.sqrt(head_dim)

        weights = torch.softmax(
            scores + attention_mask,
            dim=-1,
        )

        # eval 模式下 attention dropout 不改变数值。
        head_outputs = weights @ v

        batch_size, _, seq_length, _ = head_outputs.shape

        concatenated = (
            head_outputs
            .transpose(1, 2)
            .contiguous()
            .reshape(
                batch_size,
                seq_length,
                num_heads * head_dim,
            )
        )

        # =============================================
        # 3. 输出投影 + 第一次残差与 LayerNorm
        # =============================================
        projected = layer.attention.output.dense(concatenated)
        projected = layer.attention.output.dropout(projected)

        h = manual_layer_norm(
            x + projected,
            layer.attention.output.LayerNorm,
        )

        # =============================================
        # 4. FFN：768 → 3072 → GELU → 768
        # =============================================
        expanded = layer.intermediate.dense(h)

        # 使用当前模型配置的激活函数。
        activated = layer.intermediate.intermediate_act_fn(
            expanded
        )

        ffn_output = layer.output.dense(activated)
        ffn_after_dropout = layer.output.dropout(ffn_output)

        # =============================================
        # 5. 第二次残差与 LayerNorm
        # 注意：此处加回的是 h，而不是最初的 x。
        # =============================================
        residual = h + ffn_after_dropout

        manual_output = manual_layer_norm(
            residual,
            layer.output.LayerNorm,
        )

        # =============================================
        # 6. 与模型自带模块对照
        # =============================================

        # 对照 A：从相同的 h 开始，只验证 FFN 及其残差、归一化。
        module_ffn_output = layer.output(
            layer.intermediate(h),
            h,
        )

        # 对照 B：从相同的 x 开始，验证完整第一层。
        native_output = layer(
            x,
            attention_mask=attention_mask,
            output_attentions=False,
        )[0]

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0].tolist()
    )

    print("目标公司:", sample["company"])
    print("当前 token:", tokens[position])
    print("Device:", next(model.parameters()).device)
    print("模型是否处于 training 模式:", model.training)
    print("Activation:", layer.intermediate.intermediate_act_fn)

    print("\n=== 1. Attention 部分的 shape ===")
    print("Embedding 输出:", tuple(x.shape))
    print("Q:", tuple(q.shape))
    print("K:", tuple(k.shape))
    print("V:", tuple(v.shape))
    print("Attention weights:", tuple(weights.shape))
    print("拼接 heads 后:", tuple(concatenated.shape))
    print("第一次残差 + LayerNorm 后:", tuple(h.shape))

    print("\n=== 2. FFN 的 shape 变化 ===")
    print("FFN 输入 h:", tuple(h.shape))
    print("第一个 Linear 后:", tuple(expanded.shape))
    print("GELU 后:", tuple(activated.shape))
    print("第二个 Linear 后:", tuple(ffn_output.shape))
    print(
        "第二次残差 + LayerNorm 后:",
        tuple(manual_output.shape),
    )

    print("\n=== 3. 同一个 token 的前 8 个数 ===")
    print("FFN 输入:", first_values(h, position))
    print("扩维后:", first_values(expanded, position))
    print("GELU 后:", first_values(activated, position))
    print("降维后:", first_values(ffn_output, position))
    print(
        "完整第一层输出:",
        first_values(manual_output, position),
    )

    print("\n=== 4. GELU 如何处理前 8 个中间特征 ===")
    before = expanded[0, position, :8].tolist()
    after = activated[0, position, :8].tolist()

    for i, (a, b) in enumerate(zip(before, after)):
        print(f"维度 {i}: {a: .6f} -> {b: .6f}")

    print("\n=== 5. FFN 部分对照：使用相同的 h ===")
    compare_outputs(manual_output, module_ffn_output)

    print("\n=== 6. 完整第一层对照：使用相同的 x ===")
    compare_outputs(manual_output, native_output)


if __name__ == "__main__":
    main()