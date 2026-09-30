import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "google-bert/bert-base-chinese"
CACHE_DIR = str(PROJECT_DIR / ".cache/huggingface")

torch.manual_seed(42)

# 1. 读取一条真实 training sample
data_path = PROJECT_DIR / "data/processed/train_samples.jsonl"

with data_path.open(encoding="utf-8") as file:
    sample = json.loads(next(file))

company = sample["company"]
title = sample["title"]
news = title + "\n" + sample["text"]

# 沿用上一步的两种输入，仅用于检查模型运行
companies = [company, company]
texts = [title, news]

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
)

inputs = tokenizer(
    text=companies,
    text_pair=texts,
    truncation="only_second",
    max_length=512,
    padding=True,
    return_tensors="pt",
)

# 3. 加载预训练 BERT，并创建两类分类层
print("开始加载 BERT……")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    num_labels=2,
    id2label={0: "non_negative", 1: "negative"},
    label2id={"non_negative": 0, "negative": 1},
    use_safetensors=True,
)

# 4. 把模型和输入放到同一设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}

print("运行设备：", device)
print("input_ids shape：", tuple(inputs["input_ids"].shape))

# 5. 只进行 forward，不计算梯度，不更新参数
model.eval()

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predictions = logits.argmax(dim=1)

print("\nLogits shape：", tuple(logits.shape))
print("Logits：")
print(logits.cpu())

print("\n预测类别：", predictions.cpu().tolist())
print("原始 sample 的 label：", sample["label"])

for i, description in enumerate(["公司 + 标题", "公司 + 标题 + 正文"]):
    predicted_id = predictions[i].item()
    print(
        f"{description} → "
        f"{predicted_id} / {model.config.id2label[predicted_id]}"
    )

# 6. 两种输入都使用这条 sample 的公司级参考标签。
# 这里只演示 loss 计算，不把它们当成两条独立新闻。
labels = torch.tensor(
    [sample["label"], sample["label"]],
    dtype=torch.long,
    device=device,
)

criterion = torch.nn.CrossEntropyLoss()
loss = criterion(logits, labels)

print("\nLabels：", labels.cpu().tolist())
print("Labels shape：", tuple(labels.shape))
print("Cross-entropy loss：", loss.item())