"""Loading and preprocessing for the XL-Sum Hindi subset.

XL-Sum (Hasan et al., 2021) is a multilingual abstractive summarization
dataset built from BBC news articles and their single-sentence summaries.
We use the Hindi configuration here. See data/README.md for licensing notes.
"""

import json
import tarfile
from pathlib import Path

from textsumm.utils import clean_text, tokenize

DATASET_REPO = "csebuetnlp/xlsum"
DATASET_ARCHIVE = "data/hindi_XLSum_v2.0.tar.bz2"

# the dataset's own `datasets.load_dataset()` loading script was dropped by
# newer versions of the `datasets` library (scripts are no longer trusted by
# default), so we pull the raw jsonl archive straight from the hub instead.
SPLIT_FILENAMES = {
    "train": "hindi_train.jsonl",
    "validation": "hindi_val.jsonl",
    "test": "hindi_test.jsonl",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _extract_raw_jsonl() -> Path:
    """Download the Hindi XL-Sum archive and extract it into data/raw/, once."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    marker = RAW_DIR / SPLIT_FILENAMES["train"]
    if marker.exists():
        return RAW_DIR

    from huggingface_hub import hf_hub_download

    archive_path = hf_hub_download(DATASET_REPO, DATASET_ARCHIVE, repo_type="dataset")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(RAW_DIR, filter="data")
    return RAW_DIR


def is_usable(article: str, summary: str) -> bool:
    """Drop pairs that can't teach summarization: empty sides, or a "summary"
    at least as long as the article it summarizes.
    """
    if not article or not summary:
        return False
    return len(tokenize(summary)) < len(tokenize(article))


def download_and_clean(splits=("train", "validation", "test"), max_examples: dict | None = None):
    """Download XL-Sum Hindi and write cleaned jsonl files under data/processed/.

    max_examples optionally caps each split (e.g. {"train": 5000}) so the
    pipeline can be run end-to-end on modest hardware without pulling in the
    full ~70k training examples.
    """
    raw_dir = _extract_raw_jsonl()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    max_examples = max_examples or {}
    written = {}

    for split in splits:
        raw_path = raw_dir / SPLIT_FILENAMES[split]
        cap = max_examples.get(split)
        out_path = PROCESSED_DIR / f"{split}.jsonl"
        seen_ids: set[str] = set()
        count = dropped = 0

        with open(raw_path, encoding="utf-8") as src, open(out_path, "w", encoding="utf-8") as f:
            for line in src:
                if cap is not None and count >= cap:
                    break
                row = json.loads(line)
                article = clean_text(row["text"])
                summary = clean_text(row["summary"])
                if row["id"] in seen_ids or not is_usable(article, summary):
                    dropped += 1
                    continue
                seen_ids.add(row["id"])
                record = {"id": row["id"], "text": article, "summary": summary}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        written[split] = count
        print(f"{split}: wrote {count} examples to {out_path} (dropped {dropped})")

    return written


def load_processed(split: str, max_examples: int | None = None) -> list[dict]:
    path = PROCESSED_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py` first.")
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_examples is not None and len(examples) >= max_examples:
                break
            examples.append(json.loads(line))
    return examples


if __name__ == "__main__":
    download_and_clean()
