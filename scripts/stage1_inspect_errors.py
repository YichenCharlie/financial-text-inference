import json
import random
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
path = (
    project_dir
    / "results/stage1_baseline/validation_predictions.jsonl"
)

with path.open(encoding="utf-8") as file:
    records = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

# False positive：真实 non-negative，被预测为 negative
false_positives = [
    sample for sample in records
    if sample["label"] == 0 and sample["prediction"] == 1
]

# False negative：真实 negative，被预测为 non-negative
false_negatives = [
    sample for sample in records
    if sample["label"] == 1 and sample["prediction"] == 0
]

rng = random.Random(42)

for name, errors in [
    ("False positive：误报", false_positives),
    ("False negative：漏报", false_negatives),
]:
    print(f"\n{name}，共 {len(errors)} 条")

    selected = rng.sample(errors, min(2, len(errors)))

    for sample in selected:
        print("\nArticle ID：", sample["article_id"])
        print("目标公司：", sample["company"])
        print("标题：", sample["title"])
        print("真实 label：", sample["label"])
        print("预测 label：", sample["prediction"])
        print("正文：\n", sample["text"])