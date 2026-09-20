"""Loading and preprocessing for the XL-Sum Hindi subset.

XL-Sum (Hasan et al., 2021) is a multilingual abstractive summarization
dataset built from BBC news articles and their single-sentence summaries.
We use the Hindi configuration here. See data/README.md for licensing notes.
"""

import json
from pathlib import Path

from datasets import load_dataset

from textsumm.utils import clean_text

DATASET_NAME = "csebuetnlp/xlsum"
DATASET_CONFIG = "hindi"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def download_and_clean(splits=("train", "validation", "test"), max_examples: dict | None = None):
    """Download XL-Sum Hindi and write cleaned jsonl files under data/processed/.

    max_examples optionally caps each split (e.g. {"train": 5000}) so the
    pipeline can be run end-to-end on modest hardware without pulling in the
    full ~70k training examples.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    max_examples = max_examples or {}
    written = {}

    for split in splits:
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
        cap = max_examples.get(split)
        if cap is not None:
            dataset = dataset.select(range(min(cap, len(dataset))))

        out_path = PROCESSED_DIR / f"{split}.jsonl"
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for row in dataset:
                article = clean_text(row["text"])
                summary = clean_text(row["summary"])
                if not article or not summary:
                    continue
                f.write(json.dumps({"id": row["id"], "text": article, "summary": summary}, ensure_ascii=False) + "\n")
                count += 1
        written[split] = count
        print(f"wrote {count} examples to {out_path}")

    return written


def load_processed(split: str):
    path = PROCESSED_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


if __name__ == "__main__":
    download_and_clean()
