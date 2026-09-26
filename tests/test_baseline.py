import numpy as np

from textsumm.baseline import lead_n, oracle, pagerank, textrank
from textsumm.utils import split_sentences


def test_lead_n_returns_first_sentences(article):
    sentences = split_sentences(article)
    assert lead_n(article, n=2) == " ".join(sentences[:2])


def test_lead_n_handles_short_article():
    short = "एक ही वाक्य है।"
    assert lead_n(short, n=2) == short


def test_textrank_returns_sentences_in_original_order(article):
    sentences = split_sentences(article)
    picked = split_sentences(textrank(article, n=2))
    assert len(picked) == 2
    positions = [sentences.index(s) for s in picked]
    assert positions == sorted(positions)


def test_textrank_prefers_the_central_sentence():
    hub = "बारिश से पानी भर गया और बिजली गुल हो गई।"
    text = f"बारिश से पानी भर गया। {hub} कई इलाकों में बिजली गुल हो गई। क्रिकेट मैच कल खेला जाएगा।"
    assert textrank(text, n=1) == hub


def test_textrank_handles_short_article():
    short = "एक ही वाक्य है।"
    assert textrank(short, n=2) == short


def test_pagerank_is_a_distribution_and_favors_hubs():
    weights = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
    scores = pagerank(weights)
    assert np.isclose(scores.sum(), 1.0)
    assert scores[0] > scores[1]
    assert np.isclose(scores[1], scores[2])


def test_pagerank_handles_disconnected_nodes():
    scores = pagerank(np.zeros((3, 3)))
    assert np.allclose(scores, 1 / 3)


def test_oracle_picks_the_sentence_matching_the_reference(article):
    reference = "मौसम विभाग ने दो दिन और बारिश की चेतावनी दी"
    assert oracle(article, reference, n=3) == "मौसम विभाग ने अगले दो दिन और बारिश की चेतावनी दी है।"
