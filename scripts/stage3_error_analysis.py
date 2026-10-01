from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent.parent


PRED_PATH = (
    ROOT /
    "results/stage3_bert_validation_predictions.jsonl"
)


def load_predictions():

    data = []

    with open(
        PRED_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            data.append(
                json.loads(line)
            )

    return data



def main():

    data = load_predictions()


    false_positive = []
    false_negative = []


    for item in data:

        label = item["label"]
        pred = item["prediction"]


        if label == 0 and pred == 1:
            false_positive.append(item)


        if label == 1 and pred == 0:
            false_negative.append(item)


    print(
        "Total:",
        len(data)
    )

    print(
        "False Positive:",
        len(false_positive)
    )

    print(
        "False Negative:",
        len(false_negative)
    )


    print("\n===== False Positive Examples =====")


    for item in false_positive[:5]:

        print("\nArticle ID:",
              item["article_id"])

        print(
            "Company:",
            item["company"]
        )

        print(
            "Probability:",
            item["probability"]
        )

        print(
            "Title:",
            item["title"][:100]
        )



    print("\n===== False Negative Examples =====")


    for item in false_negative[:5]:

        print("\nArticle ID:",
              item["article_id"])

        print(
            "Company:",
            item["company"]
        )

        print(
            "Probability:",
            item["probability"]
        )

        print(
            "Title:",
            item["title"][:100]
        )


if __name__ == "__main__":
    main()