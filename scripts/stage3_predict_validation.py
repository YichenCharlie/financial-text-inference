from pathlib import Path
import json
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
)


ROOT = Path(__file__).resolve().parent.parent


MODEL_DIR = (
    ROOT /
    "models/stage3_bert_full"
)

VAL_PATH = (
    ROOT /
    "data/processed/validation_samples.jsonl"
)

OUTPUT_PATH = (
    ROOT /
    "results/stage3_bert_validation_predictions.jsonl"
)


MAX_LENGTH = 512
BATCH_SIZE = 16


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class FinancialDataset(Dataset):

    def __init__(
        self,
        path,
        tokenizer
    ):

        self.samples = []

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:
                self.samples.append(
                    json.loads(line)
                )

        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        item = self.samples[idx]

        company = item["company"]

        text = (
            item["title"]
            + "\n"
            + item["text"]
        )


        encoded = self.tokenizer(
            company,
            text,
            truncation="only_second",
            max_length=MAX_LENGTH,
        )


        encoded["labels"] = item["label"]

        return encoded



def main():

    print("Device:", DEVICE)


    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )


    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )


    model.to(DEVICE)
    model.eval()


    dataset = FinancialDataset(
        VAL_PATH,
        tokenizer
    )


    collator = DataCollatorWithPadding(
        tokenizer
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    index = 0


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:


        with torch.no_grad():

            for batch in loader:


                batch_device = {
                    k:v.to(DEVICE)
                    for k,v in batch.items()
                }


                outputs = model(
                    **batch_device
                )


                probs = torch.softmax(
                    outputs.logits,
                    dim=1
                )


                preds = torch.argmax(
                    probs,
                    dim=1
                )


                for i in range(
                    len(preds)
                ):

                    item = dataset.samples[index]


                    result = {

                        "article_id":
                            item["article_id"],

                        "company":
                            item["company"],

                        "title":
                            item["title"],

                        "label":
                            item["label"],

                        "prediction":
                            int(preds[i].cpu()),

                        "probability":
                            probs[i].cpu().tolist(),

                    }


                    f.write(
                        json.dumps(
                            result,
                            ensure_ascii=False
                        )
                        + "\n"
                    )


                    index += 1


    print(
        "Saved:",
        OUTPUT_PATH
    )



if __name__ == "__main__":
    main()