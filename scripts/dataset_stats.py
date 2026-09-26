"""Corpus statistics for the processed XL-Sum Hindi splits.

    python scripts/dataset_stats.py

Reports article/summary lengths, compression ratio, and how abstractive the
reference summaries are (share of summary n-grams absent from the article),
which bounds how well any purely extractive method can do.
"""

import json
from pathlib import Path

import numpy as np

from textsumm.data import load_processed
from textsumm.evaluate import novel_ngram_ratio
from textsumm.utils import split_sentences, tokenize

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def split_stats(examples: list[dict]) -> dict:
    article_words = np.array([len(tokenize(ex["text"])) for ex in examples])
    summary_words = np.array([len(tokenize(ex["summary"])) for ex in examples])
    sentences = np.array([len(split_sentences(ex["text"])) for ex in examples])
    return {
        "examples": len(examples),
        "article_words_mean": float(article_words.mean()),
        "article_words_p50": float(np.median(article_words)),
        "article_words_p95": float(np.percentile(article_words, 95)),
        "article_sentences_mean": float(sentences.mean()),
        "summary_words_mean": float(summary_words.mean()),
        "summary_words_p95": float(np.percentile(summary_words, 95)),
        "compression_ratio_mean": float((article_words / np.maximum(summary_words, 1)).mean()),
        "novel_unigram_ratio": float(np.mean([novel_ngram_ratio(ex["summary"], ex["text"], 1) for ex in examples])),
        "novel_bigram_ratio": float(np.mean([novel_ngram_ratio(ex["summary"], ex["text"], 2) for ex in examples])),
    }


def main():
    stats = {split: split_stats(load_processed(split)) for split in ("train", "validation", "test")}
    for split, values in stats.items():
        print(split, json.dumps(values, indent=2))
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "dataset_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
