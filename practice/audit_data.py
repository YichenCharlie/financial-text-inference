import json
import re
from collections import Counter
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent

articles = json.loads(
    (project_dir / "data/raw/train.json").read_text(encoding="utf-8")
)

# 检查重复的 article ID
id_counts = Counter(
    article["newscode"]
    for article in articles
)

# 检查标题和正文相同、但可能拥有不同 ID 的新闻
text_counts = Counter()

for article in articles:
    content = article["title"] + "\n" + article["text"]

    # 去掉空格、换行等空白字符，再比较内容
    normalized_content = re.sub(r"\s+", "", content)

    text_counts[normalized_content] += 1

duplicate_ids = sum(
    count - 1 for count in id_counts.values() if count > 1
)

duplicate_articles = sum(
    count - 1 for count in text_counts.values() if count > 1
)

print("Article 数量：", len(articles))
print("重复 ID 的额外条数：", duplicate_ids)
print("重复内容的额外条数：", duplicate_articles)