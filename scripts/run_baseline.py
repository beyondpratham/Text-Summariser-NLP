"""Score the Lead-N and TextRank baselines against the XL-Sum Hindi test set.

    python scripts/run_baseline.py --n 2 --max-samples 200
"""

import argparse
import json
from pathlib import Path

from textsumm.baseline import BASELINES
from textsumm.data import load_processed
from textsumm.evaluate import evaluate_predictions

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2, help="number of sentences to extract")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    examples = load_processed(args.split)[: args.max_samples]
    references = [ex["summary"] for ex in examples]

    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = {}

    for name, fn in BASELINES.items():
        predictions = [fn(ex["text"], n=args.n) for ex in examples]
        scores = evaluate_predictions(predictions, references)
        all_results[name] = scores
        print(f"{name}: {scores}")

    out_path = RESULTS_DIR / "baseline_scores.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
