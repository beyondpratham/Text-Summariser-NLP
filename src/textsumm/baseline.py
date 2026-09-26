"""Extractive baselines: Lead-N, TextRank, and a greedy extractive oracle.

None of these need a GPU or any fine-tuning, so they're useful both as a
sanity check on the eval pipeline and as points of comparison for the
fine-tuned abstractive model. The oracle peeks at the reference summary, so
it isn't a real system; it's the upper bound for any method that only
selects whole sentences from the article.
"""

import math
from collections import Counter

import numpy as np

from textsumm.utils import split_sentences, tokenize


def lead_n(text: str, n: int = 2) -> str:
    """Return the first n sentences of the article."""
    return " ".join(split_sentences(text)[:n])


def _tfidf_matrix(docs: list[list[str]]) -> np.ndarray:
    """L2-normalized TF-IDF rows, with the same smoothed idf as scikit-learn."""
    vocab = {tok: i for i, tok in enumerate(sorted({tok for doc in docs for tok in doc}))}
    matrix = np.zeros((len(docs), len(vocab)))
    for row, doc in enumerate(docs):
        for tok, count in Counter(doc).items():
            matrix[row, vocab[tok]] = count
    df = (matrix > 0).sum(axis=0)
    matrix *= np.log((1 + len(docs)) / (1 + df)) + 1
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def pagerank(weights: np.ndarray, damping: float = 0.85, tol: float = 1e-6, max_iter: int = 200) -> np.ndarray:
    """Weighted PageRank by power iteration. Dangling nodes link uniformly."""
    n = len(weights)
    row_sums = weights.sum(axis=1, keepdims=True)
    transition = np.where(row_sums > 0, weights / np.where(row_sums > 0, row_sums, 1), 1.0 / n)
    scores = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        updated = (1 - damping) / n + damping * transition.T @ scores
        if np.abs(updated - scores).sum() < tol:
            return updated
        scores = updated
    return scores


def textrank(text: str, n: int = 2) -> str:
    """Rank sentences by TextRank (Mihalcea & Tarau, 2004) over TF-IDF cosine similarity."""
    sentences = split_sentences(text)
    if len(sentences) <= n:
        return " ".join(sentences)

    tfidf = _tfidf_matrix([tokenize(s) for s in sentences])
    similarity = tfidf @ tfidf.T
    np.fill_diagonal(similarity, 0)
    scores = pagerank(similarity)

    # stable sort so ties go to the earlier sentence, which is the better guess for news
    ranked = sorted(range(len(sentences)), key=lambda i: -scores[i])
    return " ".join(sentences[i] for i in sorted(ranked[:n]))


def _f1(overlap: int, pred_total: int, ref_total: int) -> float:
    if overlap == 0:
        return 0.0
    precision, recall = overlap / pred_total, overlap / ref_total
    return 2 * precision * recall / (precision + recall)


def _rouge_1_2(pred_tokens: list[str], ref_unigrams: Counter, ref_bigrams: Counter) -> float:
    unigrams = Counter(pred_tokens)
    bigrams = Counter(zip(pred_tokens, pred_tokens[1:]))
    r1 = _f1(sum((unigrams & ref_unigrams).values()), len(pred_tokens), sum(ref_unigrams.values()))
    r2 = _f1(sum((bigrams & ref_bigrams).values()), max(len(pred_tokens) - 1, 0), sum(ref_bigrams.values()))
    return r1 + r2


def oracle(text: str, reference: str, n: int = 3) -> str:
    """Greedily add the sentence that most improves ROUGE-1 + ROUGE-2 F1
    against the reference, stopping when nothing helps (Nallapati et al., 2017).
    """
    sentences = split_sentences(text)
    tokenized = [tokenize(s) for s in sentences]
    ref_tokens = tokenize(reference)
    ref_unigrams = Counter(ref_tokens)
    ref_bigrams = Counter(zip(ref_tokens, ref_tokens[1:]))

    selected: list[int] = []
    best = -math.inf
    while len(selected) < n:
        candidates = []
        for i in range(len(sentences)):
            if i in selected:
                continue
            order = sorted(selected + [i])
            tokens = [tok for j in order for tok in tokenized[j]]
            candidates.append((_rouge_1_2(tokens, ref_unigrams, ref_bigrams), i))
        if not candidates:
            break
        score, index = max(candidates, key=lambda c: (c[0], -c[1]))
        if score <= best:
            break
        best = score
        selected.append(index)
    return " ".join(sentences[i] for i in sorted(selected))


BASELINES = {
    "lead_n": lead_n,
    "textrank": textrank,
}
