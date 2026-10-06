from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, BertForSequenceClassification


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/stage3_bert_full"
VAL_PATH = ROOT / "data/processed/validation_samples.jsonl"


def compare(name, actual, reference):
    difference = (actual - reference).abs()

    print(f"\n{name}")
    print(f"最大绝对差异: {difference.max().item():.8e}")
    print(
        "是否在容差内一致:",
        torch.allclose(
            actual,
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
    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model.eval()

    # 本步骤在 CPU 上验证结构；下一步才进行 GPU profiling。
    inputs = tokenizer(
        sample["company"],
        sample["title"] + "\n" + sample["text"],
        truncation="only_second",
        max_length=512,
        return_tensors="pt",
    )

    print("目标公司:", sample["company"])
    print("真实 label:", sample["label"])
    print("Device:", next(model.parameters()).device)
    print("模型是否处于 training 模式:", model.training)

    with torch.inference_mode():
        # 原生完整 forward，作为对照。
        # hidden_states 包括 Embedding 输出和各 Encoder layer 输出。
        reference = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        # 1. Embedding
        hidden = model.bert.embeddings(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
        )

        # BERT 的双向 attention mask：
        # 有效来源位置加 0，padding 来源位置加极小负数。
        padding_positions = (
            inputs["attention_mask"][:, None, None, :] == 0
        )
        attention_mask = torch.zeros(
            padding_positions.shape,
            dtype=hidden.dtype,
            device=hidden.device,
        ).masked_fill(
            padding_positions,
            torch.finfo(hidden.dtype).min,
        )

        print("\n=== 1. Embedding ===")
        print("input_ids shape:", tuple(inputs["input_ids"].shape))
        print("Embedding shape:", tuple(hidden.shape))
        compare(
            "Embedding 与原生输出对照",
            hidden,
            reference.hidden_states[0],
        )

        # 2. 依次执行 12 层。
        # 这里调用各层模块，不重复手写内部 attention 和 FFN。
        print("\n=== 2. 逐层执行 Encoder ===")
        print(
            f"{'Layer':<8}"
            f"{'Output shape':<20}"
            f"{'Max abs diff':<18}"
            f"{'Allclose'}"
        )

        for index, layer in enumerate(model.bert.encoder.layer):
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                output_attentions=False,
            )[0]

            expected = reference.hidden_states[index + 1]
            difference = (hidden - expected).abs().max().item()
            matched = torch.allclose(
                hidden,
                expected,
                atol=1e-5,
                rtol=1e-5,
            )

            print(
                f"{index + 1:<8}"
                f"{str(tuple(hidden.shape)):<20}"
                f"{difference:<18.8e}"
                f"{matched}"
            )

        # 3. 取最后一层第 0 个位置，即 [CLS]。
        cls_vector = hidden[:, 0, :]

        # 4. 显式拆开 Pooler：Linear + Tanh。
        pooler_linear = model.bert.pooler.dense(cls_vector)
        pooled = model.bert.pooler.activation(pooler_linear)

        # 5. Dropout 在 eval 模式下不改变数值，再进行分类。
        classifier_input = model.dropout(pooled)
        logits = model.classifier(classifier_input)
        probabilities = torch.softmax(logits, dim=-1)

        print("\n=== 3. 从 token 表示到分类分数 ===")
        print("最后一层全部 token:", tuple(hidden.shape))
        print("取 [CLS] 后:", tuple(cls_vector.shape))
        print("Pooler Linear 后:", tuple(pooler_linear.shape))
        print("Pooler Tanh 后:", tuple(pooled.shape))
        print("Classifier 后:", tuple(logits.shape))

        print("\n观察 [CLS] 路径的前 8 个数：")
        for name, tensor in [
            ("最终 [CLS]", cls_vector),
            ("Pooler Linear", pooler_linear),
            ("Pooler Tanh", pooled),
        ]:
            values = [
                round(value, 4)
                for value in tensor[0, :8].tolist()
            ]
            print(f"{name}: {values}")

        print("\n=== 4. 最终预测对照 ===")
        print("分步执行 logits:", logits[0].tolist())
        print("原生 forward logits:", reference.logits[0].tolist())

        compare(
            "最终 logits 对照",
            logits,
            reference.logits,
        )

        print(
            "\nProbability [non-negative, negative]:",
            probabilities[0].tolist(),
        )
        print("分步执行 prediction:", logits.argmax(dim=-1).item())
        print(
            "原生 forward prediction:",
            reference.logits.argmax(dim=-1).item(),
        )


if __name__ == "__main__":
    main()