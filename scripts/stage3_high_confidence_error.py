from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent.parent

PRED_PATH = (
    ROOT /
    "results/stage3_bert_validation_predictions.jsonl"
)


def main():

    data = []

    with open(
        PRED_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            data.append(json.loads(line))


    high_fp = []
    high_fn = []


    for item in data:

        label = item["label"]
        pred = item["prediction"]
        prob = item["probability"]


        # non-negative -> negative
        if (
            label == 0
            and pred == 1
            and prob[1] > 0.9
        ):
            high_fp.append(item)


        # negative -> non-negative
        if (
            label == 1
            and pred == 0
            and prob[0] > 0.9
        ):
            high_fn.append(item)



    print("High confidence FP:", len(high_fp))
    print("High confidence FN:", len(high_fn))


    print("\n===== High confidence FP =====")

    for item in high_fp[:10]:

        print("\nArticle:", item["article_id"])
        print("Company:", item["company"])
        print("Probability:", item["probability"])
        print("Title:", item["title"][:120])


    print("\n===== High confidence FN =====")

    for item in high_fn[:10]:

        print("\nArticle:", item["article_id"])
        print("Company:", item["company"])
        print("Probability:", item["probability"])
        print("Title:", item["title"][:120])


if __name__ == "__main__":
    main()