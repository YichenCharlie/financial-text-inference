from pathlib import Path
from collections import Counter
from datetime import datetime
import json
import math
import random
import re

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models/stage3_bert_full"
VAL_PATH = ROOT / "data/processed/validation_samples.jsonl"
PRED_PATH = ROOT / "results/stage3_bert_validation_predictions.jsonl"

MAX_LENGTH = 512
SEED = 42


def read_index(path):
    """用 article_id + company 对应原始样本和预测结果。"""
    records = {}

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            row = json.loads(line)
            key = (str(row["article_id"]), row["company"])

            if key in records:
                raise ValueError(
                    f"发现重复的 article_id/company："
                    f"{path}:{line_number}，{key}"
                )

            records[key] = row

    return records


def collect_errors(samples, predictions):
    """核对数据，然后区分 TP、TN、FP、FN。"""
    if samples.keys() != predictions.keys():
        raise ValueError(
            "Validation 与 prediction 的 article_id/company 不完全对应。"
        )

    errors = {"FP": [], "FN": []}
    counts = Counter()

    for key, pred in predictions.items():
        sample = samples[key]

        label = pred["label"]
        prediction = pred["prediction"]
        probs = pred["probability"]

        if label not in (0, 1) or prediction not in (0, 1):
            raise ValueError(f"发现非二分类 label：{key}")

        if (
            sample["label"] != label
            or sample["title"] != pred["title"]
        ):
            raise ValueError(f"原始数据与预测记录的 label/title 不一致：{key}")

        if (
            len(probs) != 2
            or not all(
                isinstance(p, (int, float))
                and math.isfinite(p)
                and 0 <= p <= 1
                for p in probs
            )
            or abs(sum(probs) - 1) > 1e-4
        ):
            raise ValueError(f"Probability 格式不正确：{key}")

        if prediction != max(range(2), key=lambda i: probs[i]):
            raise ValueError(f"Prediction 与 probability 的 argmax 不一致：{key}")

        error_type = {
            (0, 0): "TN",
            (0, 1): "FP",
            (1, 0): "FN",
            (1, 1): "TP",
        }[(label, prediction)]

        counts[error_type] += 1

        if error_type in errors:
            errors[error_type].append({
                **sample,
                "prediction": prediction,
                "probability": probs,
                "error_type": error_type,
                "confidence": probs[prediction],
            })

    return errors, counts


def select_cases(errors):
    """
    每类选 5 条高置信错误和 5 条其他错误。
    优先选择不同文章；某一组不足时从剩余样本补齐。
    """
    rng = random.Random(SEED)

    selected = []
    used_keys = set()
    used_articles = set()

    def take(pool, number):
        picked = []

        # 第一遍优先不同文章，第二遍允许同篇文章中的不同公司。
        for distinct_only in (True, False):
            for row in pool:
                if len(picked) >= number:
                    break

                key = (str(row["article_id"]), row["company"])

                if key in used_keys:
                    continue

                if distinct_only and key[0] in used_articles:
                    continue

                picked.append(row)
                used_keys.add(key)
                used_articles.add(key[0])

        return picked

    for error_type in ("FP", "FN"):
        pool = sorted(
            errors[error_type],
            key=lambda row: (
                str(row["article_id"]),
                row["company"],
            ),
        )

        if len(pool) < 10:
            raise ValueError(
                f"{error_type} 只有 {len(pool)} 条，不足 10 条。"
            )

        high = sorted(
            [row for row in pool if row["confidence"] > 0.9],
            key=lambda row: -row["confidence"],
        )

        other = [
            row for row in pool
            if row["confidence"] <= 0.9
        ]
        rng.shuffle(other)

        group = take(high, 5) + take(other, 5)
        group += take(high + other, 10 - len(group))

        selected.extend(group)

    return selected


def get_sequence_ids(encoded, sequence_number):
    """从 paired input 中取出某一段的 token IDs。"""
    return [
        encoded["input_ids"][i]
        for i, sequence_id in enumerate(encoded.sequence_ids())
        if sequence_id == sequence_number
    ]


def reconstruct_input(row, tokenizer):
    company = row["company"]
    news = row["title"] + "\n" + row["text"]

    # 与 stage3_predict_validation.py 的输入处理保持一致。
    original = tokenizer(
        company,
        news,
        truncation="only_second",
        max_length=MAX_LENGTH,
    )

    # Offset mapping 用来找到保留 token 对应的原文位置。
    traced = tokenizer(
        company,
        news,
        truncation="only_second",
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )

    for name in original:
        if original[name] != traced[name]:
            raise ValueError(f"加入 offsets 后，{name} 发生变化。")

    full = tokenizer(
        company,
        news,
        truncation=False,
        return_offsets_mapping=True,
        verbose=False,
    )

    kept_news_ids = get_sequence_ids(traced, 1)
    full_news_ids = get_sequence_ids(full, 1)

    # 确认保留的是新闻开头，且目标公司没有被截断。
    if kept_news_ids != full_news_ids[:len(kept_news_ids)]:
        raise ValueError("保留内容不是原新闻的 token 前缀，需要检查 tokenizer。")

    if get_sequence_ids(traced, 0) != get_sequence_ids(full, 0):
        raise ValueError("目标公司 segment 在截断过程中发生变化。")

    if len(original["input_ids"]) > MAX_LENGTH:
        raise ValueError("截断后的输入仍超过 MAX_LENGTH。")

    kept_positions = [
        i
        for i, sequence_id in enumerate(traced.sequence_ids())
        if sequence_id == 1
    ]

    span_end = max(
        (
            traced["offset_mapping"][i][1]
            for i in kept_positions
        ),
        default=0,
    )

    was_truncated = len(full_news_ids) > len(kept_news_ids)

    # Markdown 展示原文范围；JSONL 同时保存精确 token IDs。
    retained_news = (
        news[:span_end]
        if was_truncated
        else news
    )

    return {
        **row,
        "confidence_band": (
            ">0.9" if row["confidence"] > 0.9 else "<=0.9"
        ),
        "full_input_token_count": len(full["input_ids"]),
        "retained_input_token_count": len(original["input_ids"]),
        "removed_news_token_count": (
            len(full_news_ids) - len(kept_news_ids)
        ),
        "was_truncated": was_truncated,
        "full_news": news,
        "retained_news_source_span": retained_news,
        "source_tail_after_last_retained_token": (
            news[span_end:] if was_truncated else ""
        ),
        "model_inputs_before_padding": dict(original),
        "token_offsets": traced["offset_mapping"],
        "sequence_ids": traced.sequence_ids(),
        "news_tokens": tokenizer.convert_ids_to_tokens(kept_news_ids),
        "review": {
            "event_and_subject": "",
            "target_role": "",
            "evidence_quote": "",
            "evidence_in_retained_input": "pending",
            "possible_cause": "pending",
            "reasoning": "",
        },
    }


def text_block(value):
    """避免新闻原文中的反引号破坏 Markdown 代码块。"""
    value = str(value)

    width = max(
        [2] + [
            len(match)
            for match in re.findall(r"`+", value)
        ]
    ) + 1

    fence = "`" * width
    return f"{fence}text\n{value}\n{fence}"


def case_markdown(row, index):
    metadata = (
        f"article_id: {row['article_id']}\n"
        f"目标公司: {row['company']}\n"
        f"标题: {row['title']}\n"
        f"真实 label: {row['label']}\n"
        f"预测 label: {row['prediction']}\n"
        f"p(non-negative): {row['probability'][0]:.6f}\n"
        f"p(negative): {row['probability'][1]:.6f}\n"
        f"配对输入 tokens（截断前 / 后）: "
        f"{row['full_input_token_count']} / "
        f"{row['retained_input_token_count']}\n"
        f"新闻被移除 tokens: {row['removed_news_token_count']}\n"
        f"是否截断: {row['was_truncated']}"
    )

    return "\n\n".join([
        f"## Case {index:02d} — {row['error_type']}",
        text_block(metadata),
        "### 保留新闻的原文范围（不含独立的 company segment）",
        text_block(row["retained_news_source_span"]),
        "### 完整新闻（标题 + 正文）",
        text_block(row["full_news"]),
        "### 人工判断",
        "事件及主体：待填写。目标公司在事件中的角色：待填写。",
        "关键原文证据：待填写。证据是否进入保留输入：待核对。",
        (
            "可能原因及依据：待填写；可记录截断、关系判断、"
            "标签口径或尚不确定。仅发生截断并不等于截断造成了这条错误。"
        ),
    ])


def main():
    samples = read_index(VAL_PATH)
    predictions = read_index(PRED_PATH)

    errors, counts = collect_errors(samples, predictions)
    selected = select_cases(errors)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "当前 tokenizer 不是 fast tokenizer，无法获取原文 offsets。"
            "请把报错贴出来，先不要替换 tokenizer。"
        )

    if tokenizer.truncation_side != "right":
        raise RuntimeError(
            "当前 truncation_side 不是 right，需要先检查设置。"
        )

    reviewed = [
        reconstruct_input(row, tokenizer)
        for row in selected
    ]

    # 每次运行建立独立目录，保留之前填写的人工分析。
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = ROOT / "results/stage3_error_review" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    jsonl_path = output_dir / "review_cases.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in reviewed:
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    intro = (
        "# Stage 3 — FP/FN 人工复核材料\n\n"
        "每类目标抽取 5 条高置信（>0.9）和 5 条其他错误；"
        "高置信按分数排序，其他样本以 seed=42 打乱，优先不同文章。"
        "不足时从剩余错误补齐。这是诊断性选样，"
        "不能据此估计总体错误原因的比例。\n\n"
        "按现有预测脚本和当前保存的 tokenizer 重建输入，未重新运行模型；"
        "前提是 tokenizer 和 validation 数据没有在预测后被改动。"
        "JSONL 保存未 padding 的完整输入 IDs 与原文 offsets；"
        "Markdown 展示对应原文范围，空白或 normalization 差异"
        "不代表模型逐字符看到了原文。Company segment 完整保留；"
        "batch padding 不改变这些有效 token。错误原因留待人工填写。\n\n"
        f"Validation 数量：{len(predictions)}；"
        f"TN={counts['TN']}，FP={counts['FP']}，"
        f"FN={counts['FN']}，TP={counts['TP']}。"
    )

    cases = [
        case_markdown(row, index)
        for index, row in enumerate(reviewed, 1)
    ]

    (output_dir / "review_cases.md").write_text(
        intro + "\n\n" + "\n\n".join(cases) + "\n",
        encoding="utf-8",
    )



    print(
        f"Validation: {len(predictions)}; "
        f"TN={counts['TN']}, FP={counts['FP']}, "
        f"FN={counts['FN']}, TP={counts['TP']}"
    )

    for error_type in ("FP", "FN"):
        rows = [
            row for row in reviewed
            if row["error_type"] == error_type
        ]

        high_count = sum(
            row["confidence"] > 0.9 for row in rows
        )
        truncated_count = sum(
            row["was_truncated"] for row in rows
        )

        print(
            f"Selected {error_type}: {len(rows)}; "
            f"confidence >0.9: {high_count}; "
            f"truncated: {truncated_count}"
        )

    print(f"Saved: {output_dir}")



if __name__ == "__main__":
    main()