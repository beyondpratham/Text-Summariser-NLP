"""Score the XL-Sum authors' published mT5 model with this repo's evaluation code.

    python scripts/run_reference.py --max-samples 300

csebuetnlp/mT5_multilingual_XLSum (Hasan et al., 2021) is mT5-base fine-tuned on
all 45 XL-Sum languages on GPUs. Its reported scores use a different ROUGE
implementation, so re-scoring it here gives a like-for-like reference point.
Generation settings follow the model card. Writes results/mt5_xlsum_scores.json.
"""

import argparse
import json
import re
import time
from pathlib import Path

from textsumm.data import load_processed
from textsumm.evaluate import evaluate_predictions

MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-bertscore", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()
    examples = load_processed("test", args.max_samples)

    start = time.perf_counter()
    predictions = []
    for i in range(0, len(examples), args.batch_size):
        texts = [re.sub(r"\s+", " ", ex["text"]) for ex in examples[i : i + args.batch_size]]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.inference_mode():
            output_ids = model.generate(**batch, max_length=84, no_repeat_ngram_size=2, num_beams=4)
        predictions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        print(f"\rsummarized {len(predictions)}/{len(examples)}", end="", flush=True)
    elapsed = time.perf_counter() - start
    print()

    scores = evaluate_predictions(
        predictions,
        [ex["summary"] for ex in examples],
        [ex["text"] for ex in examples],
        bertscore=not args.no_bertscore,
    )
    scores.update(model=MODEL_NAME, split="test", articles_per_second=round(len(examples) / elapsed, 2))
    print(json.dumps(scores, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "mt5_xlsum_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()
