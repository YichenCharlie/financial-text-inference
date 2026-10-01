from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
)

import json


ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models/stage2_tiny"
DATA_PATH = ROOT / "data/processed/train_samples.jsonl"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MAX_LENGTH = 512


def load_samples(limit=64):
    """
    只读取少量 sample 做显存测试。
    不用于训练。
    """
    samples = []

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            samples.append(item)

            if len(samples) >= limit:
                break

    return samples


class TinyDataset(torch.utils.data.Dataset):

    def __init__(self, samples, tokenizer):
        self.samples = samples
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
            truncation=True,
            max_length=MAX_LENGTH,
        )

        encoded["labels"] = item["label"]

        return encoded


def test_batch(batch_size):

    print("\n====================")
    print("Testing batch size:", batch_size)
    print("====================")


    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )


    samples = load_samples(limit=64)


    dataset = TinyDataset(
        samples,
        tokenizer
    )


    collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )


    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )


    model.to(DEVICE)

    model.train()


    batch = next(iter(loader))


    batch = {
        k: v.to(DEVICE)
        for k, v in batch.items()
    }


    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()


    output = model(**batch)

    loss = output.loss

    loss.backward()


    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    peak = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )


    print(
        f"Current allocated memory: {allocated:.2f} GB"
    )

    print(
        f"Peak memory: {peak:.2f} GB"
    )


    print(
        "Input shape:",
        batch["input_ids"].shape
    )


    del model
    torch.cuda.empty_cache()



if __name__ == "__main__":


    print("Running device:", DEVICE)


    for bs in [4, 8, 16]:

        test_batch(bs)