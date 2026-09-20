"""IndicBART model/tokenizer setup for Hindi summarization.

IndicBART (Dabre et al., 2021) is an mBART-style seq2seq model pretrained on
11 Indic languages plus English, which makes it a much better fit for Hindi
than a generic multilingual model like mT5 (smaller, Devanagari-aware
vocabulary). It expects language tags in a specific place in the source and
target text - see the "<2xx>" handling below.
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "ai4bharat/IndicBART"
LANG_TAG = "<2hi>"


def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_NAME, do_lower_case=False, use_fast=False, keep_accents=True
    )


def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return model


def configure_decoder_start(model, tokenizer):
    """Point the decoder's start token at <2hi> so generation begins in Hindi.

    Training labels should NOT include the leading language tag themselves -
    the model's shift_right step injects decoder_start_token_id as the first
    decoder input automatically, so putting the tag in both places would
    just teach the model to predict its own start token twice.
    """
    lang_tag_id = tokenizer._convert_token_to_id_with_added_voc(LANG_TAG)
    model.config.decoder_start_token_id = lang_tag_id
    return model


def format_source(article_text: str) -> str:
    return f"{article_text} </s> {LANG_TAG}"


def format_target(summary_text: str) -> str:
    return f"{summary_text} </s>"
