"""Generate summaries with a trained checkpoint and score them against test references.

    python scripts/run_eval.py --checkpoint checkpoints/indicbart-hindi --max-samples 1000

Writes results/model_scores.json and results/sample_predictions.jsonl.
"""

import argparse
import json
import time
from pathlib import Path

from textsumm.data import load_processed
from textsumm.evaluate import evaluate_predictions
from textsumm.infer import GenerationSettings, Summarizer

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=1000, help="0 = whole split")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-samples-to-save", type=int, default=25)
    parser.add_argument("--no-bertscore", action="store_true")
    args = parser.parse_args()

    examples = load_processed(args.split, args.max_samples or None)
    settings = GenerationSettings(num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
    summarizer = Summarizer(args.checkpoint, device=args.device, settings=settings)

    start = time.perf_counter()
    predictions = []
    chunk = args.batch_size * 8  # sort-by-length happens within a chunk, so keep it well above batch size
    for i in range(0, len(examples), chunk):
        batch = examples[i : i + chunk]
        predictions.extend(summarizer.summarize([ex["text"] for ex in batch], batch_size=args.batch_size))
        print(f"\rsummarized {len(predictions)}/{len(examples)}", end="", flush=True)
    elapsed = time.perf_counter() - start
    print()

    scores = evaluate_predictions(
        predictions,
        [ex["summary"] for ex in examples],
        [ex["text"] for ex in examples],
        bertscore=not args.no_bertscore,
    )
    scores.update(
        split=args.split,
        checkpoint=args.checkpoint,
        device=summarizer.device,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
        articles_per_second=round(len(examples) / elapsed, 2),
    )
    print(json.dumps(scores, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "model_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    with open(RESULTS_DIR / "sample_predictions.jsonl", "w", encoding="utf-8") as f:
        for ex, pred in list(zip(examples, predictions))[: args.num_samples_to_save]:
            row = {"id": ex["id"], "text": ex["text"][:400], "reference": ex["summary"], "prediction": pred}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote scores and sample predictions to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
