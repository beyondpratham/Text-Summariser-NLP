# NLP Text Summarizer

Abstractive summarization for Hindi news articles: extractive baselines (Lead-N,
TextRank) plus a fine-tuned IndicBART model, trained and evaluated on the XL-Sum
Hindi dataset with ROUGE and BERTScore.

This started as a group project for an NLP course; see [`docs/CONTRIBUTORS.md`](docs/CONTRIBUTORS.md)
for who worked on what. The dataset, code, and everything past the initial setup
here is maintained by me.

## Why

Most summarization tooling — models, benchmarks, tutorials — targets English first.
Hindi has ~600M speakers and comparatively little of that attention. This project
builds a complete pipeline for it: data cleaning, two extractive baselines that need
no training at all, and a fine-tuned transformer, evaluated with the same metrics
(ROUGE, BERTScore) a recent shared task on Indian-language summarization used. See
[`docs/report.md`](docs/report.md) for the full write-up (motivation, related work,
methodology, results, discussion) and [`docs/ethics.md`](docs/ethics.md) for the
limitations and intended-use notes worth reading before using this on real content.

## Project layout

```
src/textsumm/
  data.py       - download + clean XL-Sum Hindi
  baseline.py   - Lead-N and TextRank extractive baselines
  model.py      - IndicBART tokenizer/model setup (language tag handling)
  train.py      - fine-tuning entry point (Seq2SeqTrainer)
  evaluate.py   - ROUGE + BERTScore scoring
  infer.py      - summarize arbitrary text from the CLI
scripts/
  download_data.py - one-shot dataset download/clean
  run_baseline.py   - score Lead-N / TextRank against the test set
  run_eval.py       - generate + score summaries from a trained checkpoint
docs/           - report, ethics notes, contributors
data/           - dataset docs (raw/processed data itself is gitignored)
results/        - score json + sample predictions from actual runs
tests/          - unit tests for utils, baselines, evaluation
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # or `source .venv/bin/activate` on Linux/Mac
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.10+. Tested on Python 3.14 (CPU-only; no CUDA required to run
the baselines or a small-scale training run — see the note on compute below).

## Usage

```bash
# 1. Download and clean the dataset (writes data/processed/*.jsonl)
python scripts/download_data.py

# 2. Score the extractive baselines (no training needed)
python scripts/run_baseline.py --n 2 --max-samples 300

# 3. Fine-tune IndicBART (defaults are sized for a CPU laptop, not a full run)
python -m textsumm.train --max-train-samples 300 --epochs 1 --output-dir checkpoints/indicbart-hindi

# 4. Evaluate the fine-tuned model against the test set
python scripts/run_eval.py --checkpoint checkpoints/indicbart-hindi --max-samples 200

# 5. Summarize your own text
python -m textsumm.infer --checkpoint checkpoints/indicbart-hindi --text "यहाँ अपना समाचार लेख लिखें..."
```

**Scaling up:** every script above takes `--max-*-samples`, `--epochs`, and
`--max-source-length` / `--max-target-length` flags. The defaults keep everything
runnable on a laptop CPU in well under an hour; on a GPU box, drop the sample caps
and bump `--epochs` to fine-tune on the full ~70.7k training set instead.

## Results

Scored on the XL-Sum Hindi test set (see [`docs/report.md`](docs/report.md) for the
exact sample sizes and full discussion — the fine-tuned model here was trained on a
small subset on a CPU, not the full dataset on a GPU, so treat these as a proof of
concept rather than a leaderboard number):

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---|---|---|---|---|
| Lead-2 | 0.235 | 0.057 | 0.167 | 0.707 |
| TextRank | 0.211 | 0.047 | 0.148 | 0.697 |
| IndicBART (fine-tuned) | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` |

Raw score files live in [`results/`](results/).

## Dataset & license

[XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum) (Hasan et al., 2021),
Hindi subset — BBC Hindi news articles paired with human-written summaries, CC
BY-NC-SA 4.0 (non-commercial). See [`data/README.md`](data/README.md) for why this
project uses XL-Sum rather than the ILSUM shared-task dataset it was originally
scoped against. This repository's own code is MIT-licensed (see [`LICENSE`](LICENSE));
that does not extend to the dataset itself.
