"""Generate summaries with a trained checkpoint and score them against test references.

    python scripts/run_eval.py --checkpoint checkpoints/indicbart-hindi --max-samples 200
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from textsumm.data import load_processed
from textsumm.evaluate import evaluate_predictions
from textsumm.infer import load, summarize

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    examples = load_processed(args.split)[: args.max_samples]
    model, tokenizer = load(args.checkpoint)

    predictions = []
    for ex in tqdm(examples, desc="summarizing"):
        predictions.append(summarize(ex["text"], model, tokenizer, max_length=args.max_length))

    references = [ex["summary"] for ex in examples]
    scores = evaluate_predictions(predictions, references)
    print(scores)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "model_scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    samples_path = RESULTS_DIR / "sample_predictions.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for ex, pred in zip(examples[:20], predictions[:20]):
            f.write(json.dumps({"text": ex["text"][:300], "reference": ex["summary"], "prediction": pred}, ensure_ascii=False) + "\n")
    print(f"wrote scores and sample predictions to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
