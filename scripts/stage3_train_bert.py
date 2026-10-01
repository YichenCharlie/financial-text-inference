from pathlib import Path
import json
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.optim import AdamW

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parent.parent


# =====================
# Paths
# =====================

PRETRAINED_MODEL = (
    ROOT
    / ".cache/huggingface/models--google-bert--bert-base-chinese"
    / "snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea"
)

TRAIN_PATH = (
    ROOT /
    "data/processed/train_samples.jsonl"
)

VAL_PATH = (
    ROOT /
    "data/processed/validation_samples.jsonl"
)

SAVE_DIR = (
    ROOT /
    "models/stage3_bert_full"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================
# Config
# =====================

MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================
# Dataset
# =====================

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



# =====================
# Evaluation
# =====================

def evaluate(model, loader):

    model.eval()

    predictions = []
    labels = []


    with torch.no_grad():

        for batch in loader:

            batch = {
                k: v.to(DEVICE)
                for k, v in batch.items()
            }


            outputs = model(**batch)


            preds = torch.argmax(
                outputs.logits,
                dim=1
            )


            predictions.extend(
                preds.cpu().tolist()
            )

            labels.extend(
                batch["labels"]
                .cpu()
                .tolist()
            )


    acc = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )


    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report":
            classification_report(
                labels,
                predictions,
                target_names=[
                    "non_negative",
                    "negative"
                ]
            ),

        "confusion_matrix":
            confusion_matrix(
                labels,
                predictions
            ).tolist()
    }



# =====================
# Main
# =====================

def main():

    print("Device:", DEVICE)


    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_MODEL,
        local_files_only=True
    )


    train_dataset = FinancialDataset(
        TRAIN_PATH,
        tokenizer
    )

    val_dataset = FinancialDataset(
        VAL_PATH,
        tokenizer
    )


    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )


    collator = DataCollatorWithPadding(
        tokenizer
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator
    )


    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=2,
        local_files_only=True
    )


    model.to(DEVICE)


    optimizer = AdamW(
        model.parameters(),
        lr=LR
    )


    best_f1 = 0


    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch+1}/{EPOCHS}"
        )

        model.train()

        total_loss = 0


        for step, batch in enumerate(
            train_loader
        ):

            batch = {
                k:v.to(DEVICE)
                for k,v in batch.items()
            }


            optimizer.zero_grad()


            outputs = model(**batch)


            loss = outputs.loss


            loss.backward()


            optimizer.step()


            total_loss += loss.item()


            if step % 100 == 0:

                print(
                    f"step {step}, "
                    f"loss={loss.item():.4f}"
                )


        avg_loss = (
            total_loss /
            len(train_loader)
        )


        print(
            "Training loss:",
            avg_loss
        )


        result = evaluate(
            model,
            val_loader
        )


        print(
            "Validation:",
            result
        )


        if result["macro_f1"] > best_f1:

            best_f1 = result["macro_f1"]

            print(
                "Saving best model..."
            )

            model.save_pretrained(
                SAVE_DIR
            )

            tokenizer.save_pretrained(
                SAVE_DIR
            )


    print(
        "\nBest Macro-F1:",
        best_f1
    )


if __name__ == "__main__":

    main()