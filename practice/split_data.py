import json
import re
import random
import hashlib
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent

articles = json.loads(
    (project_dir / "data/raw/train.json").read_text(encoding="utf-8")
)


def get_group_id(article):
    content = article["title"] + "\n" + article["text"]
    normalized = re.sub(r"\s+", "", content)

    # 相同内容得到相同的 hash（哈希值，可用作内容的指纹）
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


grouped_articles = [
    (get_group_id(article), article)
    for article in articles
]

def convert_articles(articles):
    samples = []

    # 你完成这里：
    # 1. 遍历每篇 article
    # 2. 遍历 article["institution"] 中的每个 company
    # 3. 将 company["sentiment_level"] 转成整数并判断类别
    # 4. 按上面的结构创建一个字典，加入 samples

    for article in articles:
        for company in article["institution"]:
            raw_label = int(company["sentiment_level"])
            if raw_label < 0:
                label = 1
            else:
                label = 0
    
            sample ={
                "article_id" :article["newscode"],
                "title" :article["title"],
                "text" :article["text"],
                "company" :company["ins_name"],
                "raw_label":company["sentiment_level"],
                "label":label
            }
            samples.append(sample)
    

    # 字段对应：
    # article_id ← article["newscode"]
    # title      ← article["title"]
    # text       ← article["text"]
    # company    ← company["ins_name"]
    # raw_label  ← company["sentiment_level"]
    # label      ← 负面为 1，非负面为 0

    return samples

group_ids = sorted({group_id for group_id, _ in grouped_articles})

# 固定 seed，确保相同数据可以复现相同 split
random.Random(42).shuffle(group_ids)

validation_count = max(1, round(len(group_ids) * 0.1))
validation_group_ids = set(group_ids[:validation_count])

train_articles = []
validation_articles = []

for group_id, article in grouped_articles:
    if group_id in validation_group_ids:
        validation_articles.append(article)
    else:
        train_articles.append(article)

# 检查两个集合有没有相同内容
train_groups = {get_group_id(a) for a in train_articles}
validation_groups = {get_group_id(a) for a in validation_articles}
overlap = train_groups & validation_groups

assert len(overlap) == 0
assert len(train_articles) + len(validation_articles) == len(articles)

output_dir = project_dir / "data/processed"
output_dir.mkdir(parents=True, exist_ok=True)

for filename, records in [
    ("train_articles.json", train_articles),
    ("validation_articles.json", validation_articles),
]:
    (output_dir / filename).write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

print("Training articles：", len(train_articles))
print("Validation articles：", len(validation_articles))
print("重叠 group 数量：", len(overlap))
print("已保存至：", output_dir)

# 分别转换，保持原来的 split
train_samples = convert_articles(train_articles)
validation_samples = convert_articles(validation_articles)

# 检查转换过程中有没有丢失或增加 sample
expected_count = sum(
    len(article["institution"])
    for article in articles
)

assert len(train_samples) + len(validation_samples) == expected_count

# 统计并保存两个集合
for name, samples in [
    ("train", train_samples),
    ("validation", validation_samples),
]:
    negative = sum(sample["label"] == 1 for sample in samples)
    non_negative = len(samples) - negative

    print(f"\n{name}:")
    print("Samples：", len(samples))
    print("Negative：", negative)
    print("Non-negative：", non_negative)
    print(f"Negative ratio：{negative / len(samples):.1%}")

    output_path = output_dir / f"{name}_samples.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(
                json.dumps(sample, ensure_ascii=False) + "\n"
            )

    print("保存位置：", output_path)