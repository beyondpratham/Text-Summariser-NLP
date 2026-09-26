"""Summary evaluation: ROUGE, BERTScore, bootstrap confidence intervals and
two cheap diagnostics that ROUGE alone hides:

- novel n-gram ratio: how abstractive a system is (0 = pure copying)
- unsupported-number rate: share of summaries containing a number that never
  appears in the source article, a simple proxy for factual hallucination
"""

import numpy as np
from rouge_score import rouge_scorer, tokenizers

from textsumm.utils import tokenize

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")


class HindiTokenizer(tokenizers.Tokenizer):
    """rouge_score's default tokenizer keeps only [a-z0-9], which deletes all
    Devanagari text and scores every Hindi pair as 0. This uses the
    punctuation-aware word tokenizer from textsumm.utils instead.
    """

    def tokenize(self, text):
        return tokenize(text)


def rouge_per_example(predictions: list[str], references: list[str]) -> dict[str, np.ndarray]:
    """ROUGE F1 for every (prediction, reference) pair."""
    if len(predictions) != len(references):
        raise ValueError(f"got {len(predictions)} predictions but {len(references)} references")
    scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=False, tokenizer=HindiTokenizer())
    scores = {key: np.zeros(len(predictions)) for key in ROUGE_TYPES}
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        result = scorer.score(ref, pred)
        for key in ROUGE_TYPES:
            scores[key][i] = result[key].fmeasure
    return scores


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    per_example = rouge_per_example(predictions, references)
    return {key: float(values.mean()) if len(values) else 0.0 for key, values in per_example.items()}


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap confidence interval for the mean of `values`."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def novel_ngram_ratio(summary: str, source: str, n: int = 2) -> float:
    """Fraction of the summary's n-grams that don't occur in the source."""
    summary_ngrams = _ngrams(tokenize(summary), n)
    if not summary_ngrams:
        return 0.0
    return len(summary_ngrams - _ngrams(tokenize(source), n)) / len(summary_ngrams)


def extract_numbers(text: str) -> set[str]:
    """Numeric tokens in `text`, with Devanagari digits mapped to ASCII."""
    return {tok for tok in tokenize(text) if tok.isascii() and tok.isdigit()}


def has_unsupported_number(summary: str, source: str) -> bool:
    return bool(extract_numbers(summary) - extract_numbers(source))


def compute_bertscore(predictions: list[str], references: list[str], lang: str = "hi") -> dict:
    # imported lazily: bert_score pulls in torch, which the ROUGE-only paths don't need
    from bert_score import score as bert_score

    precision, recall, f1 = bert_score(predictions, references, lang=lang, verbose=False)
    return {
        "bertscore_precision": precision.mean().item(),
        "bertscore_recall": recall.mean().item(),
        "bertscore_f1": f1.mean().item(),
    }


def evaluate_predictions(
    predictions: list[str],
    references: list[str],
    sources: list[str] | None = None,
    bertscore: bool = True,
    n_resamples: int = 1000,
) -> dict:
    """Score a system's outputs. Returns flat metrics plus 95% CIs for ROUGE."""
    per_example = rouge_per_example(predictions, references)
    results: dict = {"num_examples": len(predictions)}
    for key, values in per_example.items():
        results[key] = float(values.mean()) if len(values) else 0.0
        results[f"{key}_ci95"] = list(bootstrap_ci(values, n_resamples=n_resamples))

    lengths = [len(tokenize(p)) for p in predictions]
    results["avg_summary_words"] = float(np.mean(lengths)) if lengths else 0.0

    if sources is not None:
        results["novel_bigram_ratio"] = float(
            np.mean([novel_ngram_ratio(p, s, n=2) for p, s in zip(predictions, sources)])
        )
        results["unsupported_number_rate"] = float(
            np.mean([has_unsupported_number(p, s) for p, s in zip(predictions, sources)])
        )

    if bertscore and predictions:
        results.update(compute_bertscore(predictions, references))
    return results
