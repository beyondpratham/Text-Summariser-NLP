from textsumm.utils import clean_text, split_sentences, tokenize, truncate_words


def test_clean_text_strips_urls_and_whitespace():
    raw = "यह   एक   खबर है। http://example.com/news देखें।"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "  " not in cleaned


def test_clean_text_removes_zero_width_characters():
    assert clean_text("भारत​ में﻿") == "भारत में"


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_tokenize_keeps_devanagari_words_whole():
    assert tokenize("दिल्ली में भारी बारिश हुई।") == ["दिल्ली", "में", "भारी", "बारिश", "हुई"]


def test_tokenize_separates_punctuation_and_normalizes_digits():
    assert tokenize('"मोदी", ने कहा: २०१९ में 5.5%') == ["मोदी", "ने", "कहा", "2019", "में", "5", "5"]


def test_split_sentences_basic():
    text = "यह पहला वाक्य है। यह दूसरा वाक्य है।"
    assert len(split_sentences(text)) == 2


def test_split_sentences_handles_periods_used_as_sentence_marks():
    text = "यह पहला वाक्य है. यह दूसरा वाक्य है."
    assert len(split_sentences(text)) == 2


def test_truncate_words():
    assert truncate_words("एक दो तीन चार पांच", 3) == "एक दो तीन"


def test_truncate_words_noop_when_short_enough():
    assert truncate_words("एक दो", 5) == "एक दो"
