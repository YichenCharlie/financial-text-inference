import json
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent

train = json.loads(
    (project_dir / "data/raw/train.json").read_text(encoding="utf-8")
)


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


# 先检查一篇多公司新闻
example = convert_articles([train[7]])

print("单篇新闻转换结果：")
for sample in example:
    print(sample["company"], sample["raw_label"], sample["label"])

# 再转换整个训练文件
samples = convert_articles(train)

print("\n原始新闻篇数：", len(train))
print("转换后样本数：", len(samples))

output_path = project_dir / "data/processed/train_samples_draft.jsonl"
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as file:
    for sample in samples:
        file.write(json.dumps(sample, ensure_ascii=False) + "\n")

print("保存位置：", output_path)

negative_count = 0

for sample in samples:
    if sample["label"] == 1:
        negative_count += 1

non_negative_count = len(samples) - negative_count

print("负面样本数：", negative_count)
print("非负面样本数：", non_negative_count)
print(f"负面占比：{negative_count / len(samples):.1%}")