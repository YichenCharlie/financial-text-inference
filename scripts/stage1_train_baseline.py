import json
from collections import Counter
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# scripts 的上一级是项目根目录
project_dir = Path(__file__).resolve().parent.parent

data_dir = project_dir / "data" / "processed"
model_dir = project_dir / "models" / "stage1_baseline"
result_dir = project_dir / "results" / "stage1_baseline"


def read_jsonl(path):
    """读取 JSONL，每行转换成一个 sample。"""
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def make_text(sample):
    """构造 model 输入，不包含 label 或事件类型。"""
    return (
        f"目标公司：{sample['company']}\n"
        f"标题：{sample['title']}\n"
        f"正文：{sample['text']}"
    )


def main():
    # 1. 读取已经划分好的数据
    train = read_jsonl(data_dir / "train_samples.jsonl")
    validation = read_jsonl(data_dir / "validation_samples.jsonl")

    X_train = [make_text(sample) for sample in train]
    y_train = [sample["label"] for sample in train]

    X_validation = [make_text(sample) for sample in validation]
    y_validation = [sample["label"] for sample in validation]

    print("Training samples：", len(train), flush=True)
    print("Validation samples：", len(validation), flush=True)

    # 2. Majority baseline
    # 只根据 training set，确定数量最多的类别
    majority_label = Counter(y_train).most_common(1)[0][0]
    majority_predictions = [majority_label] * len(validation)

    print("Majority label：", majority_label, flush=True)

    # 3. 建立 TF-IDF + Logistic Regression
    # 当前版本使用目标公司、标题和完整正文
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 4),
                min_df=3,
                max_features=50000,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    # 4. Training
    # TF-IDF 和 classifier 都只在 training set 上 fit
    print("\n开始 training……", flush=True)
    model.fit(X_train, y_train)

    # 5. Validation
    # 沿用 training 学到的特征和参数
    print("开始 validation……", flush=True)
    predictions = model.predict(X_validation)

    # 6. 在同一 validation set 上比较两个方法
    metrics = {}

    for name, predicted in [
        ("majority_baseline", majority_predictions),
        ("tfidf_logistic_fulltext", predictions),
    ]:
        accuracy = accuracy_score(y_validation, predicted)

        macro_f1 = f1_score(
            y_validation,
            predicted,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )

        metrics[name] = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        }

        print(f"\n{name}")
        print(f"Accuracy：{accuracy:.4f}")
        print(f"Macro-F1：{macro_f1:.4f}")

    # 7. 创建结果目录
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整 Pipeline，包括 TF-IDF 和 classifier
    model_path = model_dir / "tfidf_logistic_fulltext.joblib"
    joblib.dump(model, model_path)

    # 保存两个方法的 validation metrics
    metrics_path = result_dir / "validation_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 保存逐条预测，方便检查错误
    predictions_path = result_dir / "validation_predictions.jsonl"

    with predictions_path.open("w", encoding="utf-8") as file:
        for sample, predicted in zip(validation, predictions):
            record = {
                **sample,
                "prediction": int(predicted),
            }
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print("\n保存完成：")
    print("Model：", model_path)
    print("Metrics：", metrics_path)
    print("Predictions：", predictions_path)


if __name__ == "__main__":
    main()