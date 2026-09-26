"""IndicBART model/tokenizer setup for Hindi summarization.

IndicBART (Dabre et al., 2022) is an mBART-style seq2seq model pretrained on
11 Indic languages plus English, which makes it a much better fit for Hindi
than a generic multilingual model like mT5 (smaller, Devanagari-aware
vocabulary). It follows mBART's sequence format:

    encoder input:   article tokens  </s> <2hi>
    labels:          summary tokens  </s> <2hi>
    decoder input:   <2hi> summary tokens </s>      (built by the model)

MBart's shift_tokens_right builds the decoder input by rotating the last
non-pad label token (<2hi>) to the front, so the language tag has to be the
final label token for training to match generation, which starts from <2hi>.
The tags are also appended *after* truncation: appending them to the raw text
and then truncating silently cuts them off every article longer than the
length limit.

transformers is imported lazily so the encoding helpers (and everything that
imports this module) work without torch installed.
"""

import re
from dataclasses import dataclass

MODEL_NAME = "ai4bharat/IndicBART"
LANG_TAG = "<2hi>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

# IndicBART registers <s>, </s> and the <2xx> tags as ordinary added tokens,
# so skip_special_tokens=True leaves them in decoded text.
_MARKUP_RE = re.compile(r"<2[a-z]{2}>|</?s>|<pad>|<unk>|\[(?:CLS|SEP|MASK)\]")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SpecialIds:
    pad: int
    bos: int
    eos: int
    lang: int


def special_ids(tokenizer) -> SpecialIds:
    ids = tokenizer.convert_tokens_to_ids([BOS_TOKEN, EOS_TOKEN, LANG_TAG])
    if tokenizer.unk_token_id in ids:
        raise ValueError(f"tokenizer is missing one of {BOS_TOKEN}, {EOS_TOKEN}, {LANG_TAG}")
    bos, eos, lang = ids
    return SpecialIds(pad=tokenizer.pad_token_id, bos=bos, eos=eos, lang=lang)


def encode(tokenizer, text: str, max_length: int) -> list[int]:
    """Token ids for `text` followed by `</s> <2hi>`, at most max_length long.

    Used for both the article and the summary: the text is truncated, never
    the suffix.
    """
    ids = special_ids(tokenizer)
    suffix = [ids.eos, ids.lang]
    body = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length - len(suffix))
    return body["input_ids"] + suffix


def clean_output(text: str) -> str:
    """Strip leftover tags from decoded model output."""
    return _SPACE_RE.sub(" ", _MARKUP_RE.sub(" ", text)).strip()


def configure_generation(model, tokenizer):
    """Set the special ids generation relies on to IndicBART's real tokens.

    The tokenizer's own bos/eos are [CLS]/[SEP], which IndicBART never uses as
    sequence boundaries, and recent versions of Trainer copy those onto the
    model config. Setting them explicitly keeps checkpoints consistent.
    """
    ids = special_ids(tokenizer)
    for config in (model.config, model.generation_config):
        config.pad_token_id = ids.pad
        config.bos_token_id = ids.bos
        config.eos_token_id = ids.eos
        config.decoder_start_token_id = ids.lang
    model.generation_config.forced_eos_token_id = ids.eos
    return model


def load_tokenizer(source: str = MODEL_NAME):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(source, do_lower_case=False, use_fast=False, keep_accents=True)


def load_model(source: str = MODEL_NAME):
    from transformers import AutoModelForSeq2SeqLM

    return AutoModelForSeq2SeqLM.from_pretrained(source)
