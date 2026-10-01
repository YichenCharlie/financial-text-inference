from pathlib import Path
import json
from transformers import AutoTokenizer
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data/processed/train_samples.jsonl"


MODEL_DIR = ROOT / "models/stage2_tiny"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)


lengths = []


with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line)

        company = sample["company"]

        text = (
            sample["title"]
            + "\n"
            + sample["text"]
        )

        encoded = tokenizer(
            company,
            text,
            truncation=False,
            add_special_tokens=True,
        )

        lengths.append(
            len(encoded["input_ids"])
        )


lengths = np.array(lengths)


print("Samples:", len(lengths))

print("\nToken length statistics")

print("Min:", lengths.min())
print("Mean:", lengths.mean())
print("Median:", np.median(lengths))

print("P90:", np.percentile(lengths,90))
print("P95:", np.percentile(lengths,95))
print("P99:", np.percentile(lengths,99))

print("Max:", lengths.max())


for limit in [128,256,512]:

    truncated = (lengths > limit).sum()

    print(
        f"\nmax_length={limit}"
    )

    print(
        "Need truncation:",
        truncated
    )

    print(
        "Ratio:",
        truncated / len(lengths)
    )