import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models/stage2_tiny"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 从本地目录加载 tokenizer 和已经微调的模型
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    use_safetensors=True,
).to(device)

# 2. 读取保存时使用的第一条检查样本
samples = json.loads(
    (MODEL_DIR / "tiny_samples.json").read_text(encoding="utf-8")
)
sample = samples[0]

# 输入处理必须与保存 reference logits 时一致
inputs = tokenizer(
    text=sample["company"],
    text_pair=sample["title"] + "\n" + sample["text"],
    truncation="only_second",
    max_length=128,
    padding="max_length",
    return_tensors="pt",
)

inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}

# 3. 在评估模式下重新计算 logits
model.eval()

with torch.no_grad():
    reloaded_logits = model(**inputs).logits.cpu()

# 4. 读取保存前的分数，进行比较
reference_logits = torch.load(
    MODEL_DIR / "reference_logits.pt",
    map_location="cpu",
    weights_only=True,
)

max_difference = (
    reloaded_logits - reference_logits
).abs().max().item()

matches = torch.allclose(
    reloaded_logits,
    reference_logits,
    atol=1e-5,
    rtol=1e-5,
)

print("运行设备：", device)
print("目标公司：", sample["company"])
print("保存前 logits：", reference_logits.tolist())
print("重新加载后 logits：", reloaded_logits.tolist())
print("最大绝对差值：", max_difference)
print("是否在容差内一致：", matches)

prediction = reloaded_logits.argmax(dim=1).item()
print("预测类别：", prediction)
print("数据集 label：", sample["label"])

if not matches:
    raise AssertionError("重加载前后的 logits 不一致，需要检查输入和运行设置。")