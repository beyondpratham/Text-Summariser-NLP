"""Extractive baselines: Lead-N and TextRank.

These don't need a GPU or any fine-tuning, so they're useful both as a
sanity check on the eval pipeline and as a point of comparison for the
fine-tuned abstractive model.
"""

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from textsumm.utils import split_sentences


def lead_n(text: str, n: int = 2) -> str:
    """Return the first n sentences of the article."""
    sentences = split_sentences(text)
    return " ".join(sentences[:n])


def textrank(text: str, n: int = 2) -> str:
    """Rank sentences by TextRank (Mihalcea & Tarau, 2004) over TF-IDF similarity."""
    sentences = split_sentences(text)
    if len(sentences) <= n:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        # happens if every sentence is empty/stopword-only after cleaning
        return " ".join(sentences[:n])

    sim_matrix = cosine_similarity(tfidf)
    np.fill_diagonal(sim_matrix, 0)

    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph, max_iter=200)

    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    top_indices = sorted(ranked[:n])
    return " ".join(sentences[i] for i in top_indices)


BASELINES = {
    "lead_n": lead_n,
    "textrank": textrank,
}
