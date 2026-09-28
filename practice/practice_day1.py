import json
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "data" / "raw" / "train.json"
train = json.loads(path.read_text(encoding="utf-8"))

# 取第一篇新闻
sample = train[7]

# 取这篇新闻中的第一家公司
company = sample["institution"][0]

# 请补上两个字段名
company_name = company["ins_name"]
raw_label = company["sentiment_level"]

print("公司名称：", company_name)
print("原始标签：", raw_label)
print("标签的数据类型：", type(raw_label).__name__)

label_number = int(raw_label)

if label_number < 0:
    label = 1
else:
    label = 0

print("转换后的标签：", label)