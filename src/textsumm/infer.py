"""Summarize a piece of Hindi text with a fine-tuned (or base) IndicBART checkpoint.

Usage:
    python -m textsumm.infer --text "..."
    python -m textsumm.infer --file path/to/article.txt --checkpoint checkpoints/indicbart-hindi
"""

import argparse

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textsumm.model import LANG_TAG, MODEL_NAME, configure_decoder_start, format_source


def load(checkpoint: str | None):
    source = checkpoint or MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(source, do_lower_case=False, use_fast=False, keep_accents=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(source)
    configure_decoder_start(model, tokenizer)
    return model, tokenizer


def summarize(text: str, model, tokenizer, max_length: int = 64, num_beams: int = 4) -> str:
    inputs = tokenizer(
        format_source(text), add_special_tokens=False, return_tensors="pt", truncation=True, max_length=512
    )
    output_ids = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc(LANG_TAG),
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a fine-tuned model dir")
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    if not args.text and not args.file:
        raise SystemExit("provide either --text or --file")

    text = args.text or open(args.file, encoding="utf-8").read()
    model, tokenizer = load(args.checkpoint)
    summary = summarize(text, model, tokenizer, max_length=args.max_length)
    print(summary)


if __name__ == "__main__":
    main()
