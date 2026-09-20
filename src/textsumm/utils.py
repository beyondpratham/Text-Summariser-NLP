"""Small shared helpers used across the data, baseline, and training modules."""

import random
import re
import unicodedata

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def clean_text(text: str) -> str:
    """Normalize whitespace/unicode noise found in scraped news text."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("​", "").replace("‌", "").replace("‍", "")
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    """Sentence-split Hindi (Devanagari) text.

    Falls back to a simple danda/period-based regex split if indicnlp isn't
    available, since the tokenizer resource download can be flaky offline.
    """
    try:
        from indicnlp.tokenize import sentence_tokenize

        return [s.strip() for s in sentence_tokenize.sentence_split(text, lang="hi") if s.strip()]
    except Exception:
        sentences = re.split(r"(?<=[।.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])
