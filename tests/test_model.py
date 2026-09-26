import pytest

from textsumm.model import clean_output, encode, special_ids

VOCAB = {"<pad>": 0, "<unk>": 1, "<s>": 64000, "</s>": 64001, "<2hi>": 64006}


class FakeTokenizer:
    """Maps each whitespace token to an id; mimics the Hugging Face call signature."""

    pad_token_id = 0
    unk_token_id = 1

    def convert_tokens_to_ids(self, tokens):
        return [VOCAB.get(t, self.unk_token_id) for t in tokens]

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [100 + len(word) for word in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_encode_appends_eos_and_language_tag():
    ids = encode(FakeTokenizer(), "एक दो तीन", max_length=16)
    assert ids[-2:] == [64001, 64006]
    assert len(ids) == 5


def test_encode_keeps_tags_when_text_is_truncated():
    long_text = " ".join(["शब्द"] * 1000)
    ids = encode(FakeTokenizer(), long_text, max_length=32)
    assert len(ids) == 32
    assert ids[-2:] == [64001, 64006]


def test_special_ids_raises_when_tag_missing():
    class NoTags(FakeTokenizer):
        def convert_tokens_to_ids(self, tokens):
            return [self.unk_token_id for _ in tokens]

    with pytest.raises(ValueError):
        special_ids(NoTags())


def test_clean_output_strips_language_tags_and_markers():
    assert clean_output("<2hi> दिल्ली में बारिश </s> [SEP]") == "दिल्ली में बारिश"
