import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_DIR / "data/processed/train_samples.jsonl"

SAMPLES_PER_CLASS = 8

tiny_samples = []
class_counts = {0: 0, 1: 0}
selected_article_ids = set()

with TRAIN_PATH.open(encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        sample = json.loads(line)
        label = sample["label"]
        article_id = sample["article_id"]

        # 同一篇文章只选一条公司 sample
        if article_id in selected_article_ids:
            continue

        # 某个类别已经选够，就跳过
        if class_counts[label] >= SAMPLES_PER_CLASS:
            continue

        tiny_samples.append(sample)
        class_counts[label] += 1
        selected_article_ids.add(article_id)

        if all(
            count == SAMPLES_PER_CLASS
            for count in class_counts.values()
        ):
            break

if len(tiny_samples) != 2 * SAMPLES_PER_CLASS:
    raise ValueError(f"没有选够样本，当前数量：{class_counts}")

print("Tiny training samples：", len(tiny_samples))
print("Non-negative：", class_counts[0])
print("Negative：", class_counts[1])
print("不同 Article 数量：", len(selected_article_ids))

print("\n前 4 条 sample：")
for sample in tiny_samples[:4]:
    print(
        sample["article_id"],
        sample["company"],
        sample["label"],
    )

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.manual_seed(42)

MODEL_NAME = "google-bert/bert-base-chinese"
CACHE_DIR = str(PROJECT_DIR / ".cache/huggingface")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 将 16 条 sample 转成模型输入
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
)

companies = [sample["company"] for sample in tiny_samples]
news = [
    sample["title"] + "\n" + sample["text"]
    for sample in tiny_samples
]
labels = torch.tensor(
    [sample["label"] for sample in tiny_samples],
    dtype=torch.long,
)

encoded = tokenizer(
    text=companies,
    text_pair=news,
    truncation="only_second",
    max_length=128,
    padding=True,
    return_tensors="pt",
)

# 2. 将输入和标签配对，再分成小批次
dataset = TensorDataset(
    encoded["input_ids"],
    encoded["attention_mask"],
    encoded["token_type_ids"],
    labels,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

# 3. 加载模型与 optimizer
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    num_labels=2,
    id2label={0: "non_negative", 1: "negative"},
    label2id={"non_negative": 0, "negative": 1},
    use_safetensors=True,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01,
)

# 4. 固定模式下检查这 16 条样本，方便比较训练前后
def inspect_tiny_set():
    model.eval()
    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, token_type_ids, batch_labels = [
                tensor.to(device) for tensor in batch
            ]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=batch_labels,
            )

            total_loss += outputs.loss.item() * len(batch_labels)
            predictions = outputs.logits.argmax(dim=1)
            total_correct += (predictions == batch_labels).sum().item()

    return total_loss / len(dataset), total_correct / len(dataset)


before_loss, before_accuracy = inspect_tiny_set()
weight_before = model.classifier.weight.detach().clone()

print(f"\n训练前：loss={before_loss:.4f}, accuracy={before_accuracy:.4f}")

# 5. 正式进行参数更新
for epoch in range(3):
    model.train()

    for step, batch in enumerate(loader, start=1):
        input_ids, attention_mask, token_type_ids, batch_labels = [
            tensor.to(device) for tensor in batch
        ]

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=batch_labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch + 1}/3 | "
            f"Batch {step}/{len(loader)} | "
            f"loss={loss.item():.4f}"
        )

    eval_loss, accuracy = inspect_tiny_set()
    print(
        f"Epoch {epoch + 1} 结束："
        f"tiny-set loss={eval_loss:.4f}, accuracy={accuracy:.4f}"
    )

weight_change = (
    model.classifier.weight.detach() - weight_before
).abs().max().item()

print("\n分类层 weight 的最大绝对变化：", weight_change)

# 6. 保存微调后的模型、tokenizer 和本次样本
output_dir = PROJECT_DIR / "models/stage2_tiny"
output_dir.mkdir(parents=True, exist_ok=True)

model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

(output_dir / "tiny_samples.json").write_text(
    json.dumps(tiny_samples, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# 保存第一条 sample 的分数，供新进程加载后对照
check_inputs = tokenizer(
    text=tiny_samples[0]["company"],
    text_pair=(
        tiny_samples[0]["title"] + "\n" + tiny_samples[0]["text"]
    ),
    truncation="only_second",
    max_length=128,
    padding="max_length",
    return_tensors="pt",
)
check_inputs = {
    name: tensor.to(device)
    for name, tensor in check_inputs.items()
}

model.eval()
with torch.no_grad():
    reference_logits = model(**check_inputs).logits.cpu()

torch.save(reference_logits, output_dir / "reference_logits.pt")

print("\n模型保存位置：", output_dir)
print("重加载检查用 logits：", reference_logits.tolist())