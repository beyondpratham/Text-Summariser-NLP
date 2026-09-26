# Hindi News Summarizer

[![tests](https://github.com/beyondpratham/Text-Summariser-NLP/actions/workflows/tests.yml/badge.svg)](https://github.com/beyondpratham/Text-Summariser-NLP/actions/workflows/tests.yml)

End-to-end summarization for Hindi news: a data pipeline over the 88k-article
XL-Sum Hindi corpus, extractive baselines, a fine-tuned IndicBART model, an
evaluation harness with confidence intervals and hallucination checks, and a
REST API that serves it all.

Most summarization tooling is built and benchmarked for English first. Hindi has
~600M speakers and far less of that attention, and several standard tools break
on Devanagari text without any error message (see [What broke on Hindi](#what-broke-on-hindi)).

## Highlights

- **Data pipeline** that downloads, cleans (Unicode normalization, zero-width
  characters, URLs), de-duplicates, and filters 88,471 article/summary pairs.
- **Baselines built from scratch in NumPy**: Lead-N, TextRank (TF-IDF + power-iteration
  PageRank), and a greedy extractive oracle for the upper bound.
- **IndicBART fine-tuning** with Hugging Face `Seq2SeqTrainer`: in-loop ROUGE
  validation, best-checkpoint selection, early stopping, mixed precision, label smoothing.
- **Batched inference** with length-sorted batching to minimize padding, beam search, and
  n-gram repetition blocking. Runs on CUDA, Apple MPS, or CPU.
- **Evaluation** beyond a single ROUGE number: 95% bootstrap confidence intervals,
  abstractiveness (novel n-gram ratio), and an unsupported-number rate that flags
  summaries containing figures absent from the source article.
- **FastAPI service** with request validation, batch requests, lazy thread-safe model
  loading, and a health check. Ships as a Docker image.
- **38 unit tests** and CI (lint + tests on Python 3.10/3.12). The test suite runs
  without torch because the model code is tested against a fake tokenizer.

## Results

XL-Sum Hindi **test set, all 8,847 articles**. ROUGE F1 uses a Devanagari-aware
tokenizer; ± is the half-width of the 95% bootstrap confidence interval.

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg. words |
|---|---|---|---|---|
| Lead-1 | 22.75 ± 0.22 | 5.31 ± 0.14 | 17.23 ± 0.18 | 23.1 |
| Lead-2 | **25.12** ± 0.19 | 6.14 ± 0.12 | 17.31 ± 0.14 | 44.2 |
| Lead-3 | 24.16 ± 0.16 | **6.22** ± 0.11 | 16.10 ± 0.12 | 65.1 |
| TextRank (1 sentence) | 24.28 ± 0.21 | 5.23 ± 0.14 | **17.46** ± 0.17 | 30.9 |
| *Extractive oracle (upper bound)* | *38.47* ± 0.19 | *14.83* ± 0.20 | *26.09* ± 0.20 | 38.4 |

Scores ×100. Raw output: [`results/baseline_scores.json`](results/baseline_scores.json).

The oracle picks the sentences that best match the reference, so it is the
ceiling for any system that only copies sentences from the article. 73% of the
reference summaries' bigrams never appear in the article
([`results/dataset_stats.json`](results/dataset_stats.json)), so even that ceiling
is low. That gap is what the abstractive model is for.

**Fine-tuned IndicBART:** the model has not yet been retrained with the fixes below. The
earlier proof-of-concept (300 training examples, 1 CPU epoch, before the fixes)
reached ROUGE-1 0.230 on a 200-article sample and did not beat Lead-N. Full
retraining on a GPU uses the command in [Training at full scale](#training-at-full-scale).

## What broke on Hindi

These bugs all failed silently. Each one produced plausible-looking output and a
lower score, with no error.

| Problem | Effect | Fix |
|---|---|---|
| `\w` regex doesn't match Devanagari vowel signs. scikit-learn's default TF-IDF tokenizer returns **zero tokens** for "दिल्ली में भारी बारिश हुई।" | TextRank's similarity graph was mostly empty, so it ranked sentences close to randomly | Unicode-category tokenizer ([`utils.tokenize`](src/textsumm/utils.py)) shared by TextRank and ROUGE |
| `rouge_score` keeps only `[a-z0-9]` | Every Hindi pair scores 0 | Custom tokenizer that also splits off the danda (।), so `हुई।` matches `हुई` |
| The `</s> <2hi>` suffix was appended to the article *before* truncating to 384 tokens | The language tag and EOS were cut from nearly every training article | Truncate the text, then append the suffix ([`model.encode`](src/textsumm/model.py)) |
| Same truncation issue on labels | Long summaries lost EOS, so the model never learned to stop. Every prediction ran to the length cap | Same fix |
| mBART builds the decoder input by rotating the *last* label token to the front | With labels `summary </s>`, training decoded from `</s>` but inference started from `<2hi>` | mBART's target format `summary </s> <2hi>` |
| IndicBART registers `<2hi>` and `</s>` as non-special tokens | `skip_special_tokens` left `<2hi>` at the start of every prediction | Strip tags after decoding and pin generation ids explicitly |

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # or: pip install -e ".[serve]" for the API without training deps

python scripts/download_data.py      # download + clean XL-Sum Hindi -> data/processed/
python scripts/dataset_stats.py      # corpus statistics
python scripts/run_baseline.py       # score Lead-N / TextRank / oracle on the full test set
pytest                               # unit tests
```

### Serve the API

```bash
# extractive methods work with no model; set a checkpoint to enable the abstractive one
TEXTSUMM_CHECKPOINT=checkpoints/indicbart-hindi uvicorn textsumm.api:app --port 8000

curl -X POST localhost:8000/summarize -H "Content-Type: application/json" \
  -d '{"texts": ["दिल्ली में आज भारी बारिश हुई। सड़कों पर पानी भर गया।"], "method": "model"}'
# {"summaries": ["..."], "method": "model", "latency_ms": 412.7}
```

`method` is one of `model`, `lead`, `textrank`. Up to 32 texts per request.
Interactive docs are at `/docs`.

With Docker:

```bash
docker build -t hindi-summarizer .
docker run -p 8000:8000 -v "$(pwd)/checkpoints/indicbart-hindi:/model" hindi-summarizer
```

### Training at full scale

```bash
# single GPU (e.g. a free Kaggle/Colab T4), full 70.7k training set
python -m textsumm.train --max-train-samples 0 --epochs 3 --batch-size 16 --grad-accum 2 \
    --output-dir checkpoints/indicbart-hindi

python scripts/run_eval.py --checkpoint checkpoints/indicbart-hindi --max-samples 0
```

For a quick CPU smoke test, use `--max-train-samples 300 --eval-steps 50`.

## Project layout

```
src/textsumm/
  data.py        download, clean, de-duplicate, filter
  utils.py       Devanagari-aware tokenizer, sentence splitter, text cleaning
  baseline.py    Lead-N, TextRank (NumPy TF-IDF + PageRank), extractive oracle
  model.py       IndicBART sequence formatting and generation config
  train.py       fine-tuning with in-loop ROUGE, early stopping, mixed precision
  infer.py       batched Summarizer class + CLI
  evaluate.py    ROUGE, BERTScore, bootstrap CIs, abstractiveness, unsupported numbers
  api.py         FastAPI service
scripts/         download_data, dataset_stats, run_baseline, run_eval
tests/           unit tests (no torch or network needed)
docs/            project report and ethics notes
results/         score files from actual runs
```

## Dataset and license

[XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum) (Hasan et al., 2021),
Hindi subset: BBC Hindi articles with summaries written by BBC editors, licensed
CC BY-NC-SA 4.0 (non-commercial). See [`data/README.md`](data/README.md). The
code in this repository is MIT-licensed ([`LICENSE`](LICENSE)). That license
does not cover the dataset.

The full write-up (related work, method, discussion) is in
[`docs/report.md`](docs/report.md). Limitations and intended use are in
[`docs/ethics.md`](docs/ethics.md). This project started as a course project;
see [`docs/CONTRIBUTORS.md`](docs/CONTRIBUTORS.md).
