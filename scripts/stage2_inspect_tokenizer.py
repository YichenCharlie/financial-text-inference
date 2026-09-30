import json
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "google-bert/bert-base-chinese"

# 先用 128 方便观察，不代表最终训练长度。
MAX_LENGTH = 512

# 1. 读取第一条 training sample
data_path = PROJECT_DIR / "data/processed/train_samples.jsonl"

with data_path.open(encoding="utf-8") as file:
    sample = json.loads(next(file))

company = sample["company"]
news = sample["title"] + "\n" + sample["text"]

print("目标公司：", company)
print("标题：", sample["title"])
print("Label：", sample["label"])
print("正文前 150 个字符：", sample["text"][:150])

# 2. 加载 tokenizer；缓存放在项目的数据盘目录
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=str(PROJECT_DIR / ".cache/huggingface"),
)

# 3. 分别查看公司和新闻的 token 数量
company_tokens = tokenizer.tokenize(company)
news_tokens = tokenizer.tokenize(news)
special_count = tokenizer.num_special_tokens_to_add(pair=True)

full_length = len(company_tokens) + len(news_tokens) + special_count

print("\n公司 tokens：", company_tokens)
print("公司 token 数：", len(company_tokens))
print("新闻 token 数：", len(news_tokens))
print("特殊 token 数：", special_count)
print("截断前总 token 数：", full_length)

if len(company_tokens) + special_count >= MAX_LENGTH:
    raise ValueError("公司名称过长，请增大 MAX_LENGTH 后再运行。")

# 4. 第一段是公司，第二段是新闻；超长时只截断新闻
encoded = tokenizer(
    text=company,
    text_pair=news,
    truncation="only_second",
    max_length=MAX_LENGTH,
    padding=False,
    return_tensors="pt",
)

# 5. 查看三个输入 tensor 的 shape
print("\n模型输入的 shape：")
for name, tensor in encoded.items():
    print(f"{name}：{tuple(tensor.shape)}")

input_ids = encoded["input_ids"][0].tolist()
tokens = tokenizer.convert_ids_to_tokens(input_ids)
type_ids = encoded["token_type_ids"][0].tolist()
mask = encoded["attention_mask"][0].tolist()

print("\n处理后 token 数：", len(input_ids))
print("被截去的 token 数：", full_length - len(input_ids))

# 6. 将前 40 个位置逐项对应起来
print("\n前 40 个位置：")
print(f"{'位置':<6} {'token':<14} {'ID':<8} {'段编号':<8} mask")

for i in range(min(40, len(input_ids))):
    print(
        f"{i:<6} {tokens[i]:<14} "
        f"{input_ids[i]:<8} {type_ids[i]:<8} {mask[i]}"
    )

print("\n保留下来的末尾 20 个 tokens：")
print(tokens[-20:])


# 7. 用同一家公司、两种新闻长度观察 padding
# 第一条仅用标题，第二条用标题 + 正文。
# 这只是输入格式演示，不用于比较模型效果。
batch_companies = [company, company]
batch_news = [sample["title"], news]

batch_encoded = tokenizer(
    text=batch_companies,
    text_pair=batch_news,
    truncation="only_second",
    max_length=MAX_LENGTH,
    padding=True,
    return_tensors="pt",
)

print("\n=== Batch 与 padding 实验 ===")

for name, tensor in batch_encoded.items():
    print(f"{name}：{tuple(tensor.shape)}")

for i in range(len(batch_companies)):
    mask = batch_encoded["attention_mask"][i]

    real_length = int(mask.sum().item())
    total_length = mask.numel()
    padding_length = total_length - real_length

    print(f"\n第 {i + 1} 条输入：")
    print("真实 token 数（包含特殊 tokens）：", real_length)
    print("Padding 数：", padding_length)
    print("补齐后总长度：", total_length)

    tail_ids = batch_encoded["input_ids"][i, -15:].tolist()
    tail_tokens = tokenizer.convert_ids_to_tokens(tail_ids)

    print("末尾 15 个 tokens：", tail_tokens)
    print("末尾 15 个 mask：", mask[-15:].tolist())