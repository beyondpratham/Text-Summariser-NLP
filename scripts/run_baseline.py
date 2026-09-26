"""Score the extractive baselines against the XL-Sum Hindi test set.

    python scripts/run_baseline.py                       # full test set, ROUGE + BERTScore
    python scripts/run_baseline.py --max-samples 500 --no-bertscore

Writes results/baseline_scores.json.
"""

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from textsumm.baseline import lead_n, oracle, textrank
from textsumm.data import load_processed
from textsumm.evaluate import evaluate_predictions

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = whole split")
    parser.add_argument("--lead", type=int, nargs="+", default=[1, 2, 3], help="Lead-N sizes to score")
    parser.add_argument("--textrank-n", type=int, default=1)
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "baseline_scores.json"))
    args = parser.parse_args()

    examples = load_processed(args.split, args.max_samples or None)
    sources = [ex["text"] for ex in examples]
    references = [ex["summary"] for ex in examples]

    systems = {f"lead_{n}": lambda ex, n=n: lead_n(ex["text"], n) for n in args.lead}
    systems[f"textrank_{args.textrank_n}"] = lambda ex: textrank(ex["text"], args.textrank_n)
    systems["oracle"] = lambda ex: oracle(ex["text"], ex["summary"])

    results = {"split": args.split, "num_examples": len(examples), "systems": {}}
    for name, system in systems.items():
        start = time.perf_counter()
        predictions = [system(ex) for ex in tqdm(examples, desc=name)]
        scores = evaluate_predictions(predictions, references, sources, bertscore=not args.no_bertscore)
        scores["seconds"] = round(time.perf_counter() - start, 1)
        results["systems"][name] = scores
        print(f"{name}: R1={scores['rouge1']:.4f} R2={scores['rouge2']:.4f} RL={scores['rougeL']:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
