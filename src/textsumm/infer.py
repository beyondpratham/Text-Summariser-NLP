"""Summarize Hindi text with a fine-tuned (or base) IndicBART checkpoint.

Usage:
    python -m textsumm.infer --checkpoint checkpoints/indicbart-hindi --text "..."
    python -m textsumm.infer --checkpoint checkpoints/indicbart-hindi --file article.txt
"""

import argparse
from dataclasses import dataclass

from textsumm.model import (
    MODEL_NAME,
    clean_output,
    configure_generation,
    encode,
    load_model,
    load_tokenizer,
    special_ids,
)
from textsumm.utils import clean_text


@dataclass
class GenerationSettings:
    max_source_length: int = 512
    max_new_tokens: int = 80
    num_beams: int = 4
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0


def pick_device(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Summarizer:
    """Loads a checkpoint once and summarizes texts in padded batches."""

    def __init__(self, checkpoint: str | None = None, device: str | None = None, settings=None):
        self.checkpoint = checkpoint or MODEL_NAME
        self.device = pick_device(device)
        self.settings = settings or GenerationSettings()
        self.tokenizer = load_tokenizer(self.checkpoint)
        self.model = configure_generation(load_model(self.checkpoint), self.tokenizer)
        self.model.to(self.device).eval()
        self._ids = special_ids(self.tokenizer)

    def summarize(self, texts: list[str], batch_size: int = 8) -> list[str]:
        import torch

        encoded = [encode(self.tokenizer, clean_text(t), self.settings.max_source_length) for t in texts]
        # longest-first batching keeps padding (and wasted compute) to a minimum
        order = sorted(range(len(texts)), key=lambda i: -len(encoded[i]))
        summaries = [""] * len(texts)

        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            batch = self.tokenizer.pad({"input_ids": [encoded[i] for i in batch_idx]}, return_tensors="pt").to(
                self.device
            )
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **batch,
                    max_new_tokens=self.settings.max_new_tokens,
                    num_beams=self.settings.num_beams,
                    no_repeat_ngram_size=self.settings.no_repeat_ngram_size,
                    length_penalty=self.settings.length_penalty,
                    early_stopping=True,
                    pad_token_id=self._ids.pad,
                    bos_token_id=self._ids.bos,
                    eos_token_id=self._ids.eos,
                    decoder_start_token_id=self._ids.lang,
                )
            decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for i, text in zip(batch_idx, decoded):
                summaries[i] = clean_output(text)
        return summaries

    def __call__(self, text: str) -> str:
        return self.summarize([text])[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", type=str)
    source.add_argument("--file", type=str)
    parser.add_argument("--checkpoint", type=str, default=None, help="fine-tuned model dir (default: base IndicBART)")
    parser.add_argument("--device", type=str, default=None, help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--num-beams", type=int, default=4)
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text

    settings = GenerationSettings(max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
    summarizer = Summarizer(args.checkpoint, device=args.device, settings=settings)
    print(summarizer(text))


if __name__ == "__main__":
    main()
