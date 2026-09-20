from textsumm.utils import clean_text, split_sentences, truncate_words


def test_clean_text_strips_urls_and_whitespace():
    raw = "यह   एक   खबर है। http://example.com/news देखें।"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "  " not in cleaned


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_split_sentences_basic():
    text = "यह पहला वाक्य है। यह दूसरा वाक्य है।"
    sentences = split_sentences(text)
    assert len(sentences) == 2


def test_truncate_words():
    text = "एक दो तीन चार पांच"
    assert truncate_words(text, 3) == "एक दो तीन"


def test_truncate_words_noop_when_short_enough():
    text = "एक दो"
    assert truncate_words(text, 5) == text
