"""Small shared helpers used across the data, baseline, evaluation and training modules."""

import random
import re
import unicodedata

import numpy as np

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"))
_URL_RE = re.compile(r"http\S+|www\.\S+")
_SPACE_RE = re.compile(r"\s+")
_SENTENCE_END_RE = re.compile(r"(?<=[।॥.!?])\s+")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def clean_text(text: str | None) -> str:
    """Normalize whitespace/unicode noise found in scraped news text."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_ZERO_WIDTH)
    text = _URL_RE.sub("", text)
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens, dropping punctuation and symbols.

    Regex word patterns like ``\\w+`` don't work for Hindi: Python's ``\\w``
    doesn't match Devanagari vowel signs (matras), so words get split in the
    middle or dropped entirely. Filtering by Unicode category keeps every
    word intact and still separates the danda (।) and other punctuation from
    the word before it, so "हुई।" and "हुई" count as the same token.
    """
    text = unicodedata.normalize("NFC", text).lower().translate(_DEVANAGARI_DIGITS)
    chars = (" " if unicodedata.category(ch)[0] in "PSZ" else ch for ch in text)
    return "".join(chars).split()


def split_sentences(text: str) -> list[str]:
    """Sentence-split Hindi (Devanagari) text.

    Uses indic-nlp-library's splitter, which knows about Hindi abbreviations,
    and falls back to a danda/punctuation regex if it isn't installed.
    """
    try:
        from indicnlp.tokenize import sentence_tokenize
    except ImportError:
        sentences = _SENTENCE_END_RE.split(text)
    else:
        sentences = sentence_tokenize.sentence_split(text, lang="hi")
    return [s.strip() for s in sentences if s.strip()]


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])
