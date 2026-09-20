from textsumm.baseline import lead_n, textrank
from textsumm.utils import split_sentences

ARTICLE = (
    "दिल्ली में आज भारी बारिश हुई। सड़कों पर पानी भर गया। "
    "यातायात पुलिस ने लोगों को सतर्क रहने को कहा। कई इलाकों में बिजली गुल हो गई। "
    "मौसम विभाग ने अगले दो दिन और बारिश की चेतावनी दी है।"
)


def test_lead_n_returns_requested_sentence_count():
    summary = lead_n(ARTICLE, n=2)
    assert summary.count("।") <= 2
    assert summary.startswith("दिल्ली")


def test_lead_n_handles_short_article():
    short = "एक ही वाक्य है।"
    assert lead_n(short, n=2) == short


def test_textrank_returns_subset_of_sentences():
    summary = textrank(ARTICLE, n=2)
    original_sentences = set(split_sentences(ARTICLE))
    picked = split_sentences(summary)
    assert len(picked) == 2
    assert all(sentence in original_sentences for sentence in picked)


def test_textrank_handles_short_article():
    short = "एक ही वाक्य है।"
    assert textrank(short, n=2) == short
