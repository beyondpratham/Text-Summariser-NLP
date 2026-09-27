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
  validation, best-checkpoint selection, early stopping, mixed precision, label smoothing,
  warm-starting from any IndicBART-family checkpoint, and resuming on unseen data.
  The fine-tuned model **beats all extractive baselines by 4.5 ROUGE-L**, with
  non-overlapping 95% CIs.
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

All systems are scored on the **same 1,000 XL-Sum Hindi test articles** (the first
1,000 of the 8,847-article test split). ROUGE F1 uses a Devanagari-aware tokenizer;
± is the half-width of the 95% bootstrap confidence interval. Scores ×100.

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Words |
|---|---|---|---|---|---|
| Lead-1 | 22.15 ± 0.64 | 4.75 ± 0.37 | 16.62 ± 0.51 | 70.72 | 22.9 |
| Lead-2 | 24.67 ± 0.57 | 5.72 ± 0.35 | 17.01 ± 0.43 | 70.68 | 43.8 |
| Lead-3 | 23.77 ± 0.49 | 5.80 ± 0.32 | 15.92 ± 0.35 | 70.37 | 64.5 |
| TextRank (1 sentence) | 24.11 ± 0.58 | 4.97 ± 0.36 | 17.04 ± 0.48 | 70.53 | 30.5 |
| IndicBART-SS, zero-shot | 18.80 ± 0.64 | 4.70 ± 0.40 | 15.38 ± 0.58 | 69.62 | 10.1 |
| **IndicBART-SS, fine-tuned (this repo)** | **26.76** ± 0.73 | **7.46** ± 0.53 | **21.57** ± 0.65 | **73.05** | 17.7 |
| *Extractive oracle (upper bound)* | *38.20* ± 0.53 | *14.63* ± 0.55 | *25.82* ± 0.54 | *74.27* | 38.1 |

IndicBART-SS is [`ai4bharat/MultiIndicSentenceSummarization`](https://huggingface.co/ai4bharat/MultiIndicSentenceSummarization),
IndicBART already fine-tuned on Indic sentence summarization. Raw scores and sample
outputs are in [`results/`](results/).

- **Fine-tuning beats every extractive baseline**, with no overlap in confidence
  intervals: ROUGE-L +4.5 and BERTScore +2.3 over the best baseline. Lead-N is
  hard to beat on news, because journalists put the key facts first.
- **Fine-tuning is what makes the difference.** The same checkpoint before fine-tuning
  writes 10-word headlines and scores ROUGE-1 18.8. After fine-tuning it scores 26.8
  (+8.0).
- **The model paraphrases.** 47% of its bigrams do not appear in the article,
  against 0% for the extractive systems.
- **The extractive ceiling is low.** Even the oracle, which chooses sentences
  while looking at the reference, reaches only 38.2 ROUGE-1. That's because 73%
  of reference bigrams never appear in the article
  ([`results/dataset_stats.json`](results/dataset_stats.json)).

**Training budget.** The fine-tuned model was trained on a laptop CPU (Intel
i5-8250U, 8 GB RAM, no GPU). It saw 1,600 training articles (2.3% of the
training set), took 200 optimizer steps, and ran for 3.5 hours with 256-token
inputs. The best checkpoint was chosen by validation ROUGE-L. The numbers above
are therefore a lower bound for this pipeline, and training on the full set with
a GPU is the obvious next step ([Training at full scale](#training-at-full-scale)).

**Known weaknesses.** Summaries are shorter than the references (18 vs. 27
words), which costs recall. 3.1% contain a number that is not in the article. Manual
review also finds factual slips, such as placing Malegaon in Madhya Pradesh
instead of Maharashtra. Treat outputs as drafts for human review
([`docs/ethics.md`](docs/ethics.md)).

On the full 8,847-article test set, the extractive baselines score within 0.6
points of the subset above (Lead-2: 25.12 / 6.14 / 17.31;
[`results/baseline_scores.json`](results/baseline_scores.json)), so the subset is
representative.

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
TEXTSUMM_CHECKPOINT=checkpoints/indicbart-xlsum-hi uvicorn textsumm.api:app --port 8000

curl -X POST localhost:8000/summarize -H "Content-Type: application/json" \
  -d '{"texts": ["दिल्ली में आज भारी बारिश हुई। सड़कों पर पानी भर गया।"], "method": "model"}'
# {"summaries": ["..."], "method": "model", "latency_ms": 412.7}
```

`method` is one of `model`, `lead`, `textrank`. Up to 32 texts per request.
Interactive docs are at `/docs`.

With Docker:

```bash
docker build -t hindi-summarizer .
docker run -p 8000:8000 -v "$(pwd)/checkpoints/indicbart-xlsum-hi:/model" hindi-summarizer
```

### Reproduce the reported model (CPU, about 3.5 hours)

```bash
python -m textsumm.train --model-name ai4bharat/MultiIndicSentenceSummarization     --freeze-embeddings --max-train-samples 1600 --max-eval-samples 64     --max-source-length 256 --max-target-length 64 --batch-size 4 --grad-accum 2     --lr 1e-4 --eval-steps 50 --output-dir checkpoints/indicbart-xlsum-hi

python scripts/run_eval.py --checkpoint checkpoints/indicbart-xlsum-hi --max-samples 1000     --max-source-length 256 --output-name indicbart_finetuned
```

`--freeze-embeddings` keeps the shared 64k x 1024 vocabulary matrix fixed, which
saves about 0.8 GB of optimizer state and fits training into 8 GB of RAM. To keep
training on examples the model hasn't seen, pass `--model-name <checkpoint>
--train-offset 1600`. Checkpoints are about 1 GB and are not committed.

### Training at full scale

```bash
# single GPU (e.g. a free Kaggle/Colab T4), full 70.7k training set, full article context
python -m textsumm.train --model-name ai4bharat/MultiIndicSentenceSummarization     --max-train-samples 0 --epochs 3 --batch-size 16 --grad-accum 2     --output-dir checkpoints/indicbart-xlsum-hi-full

python scripts/run_eval.py --checkpoint checkpoints/indicbart-xlsum-hi-full --max-samples 0
```

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
scripts/         download_data, dataset_stats, run_baseline, run_eval, run_reference
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
