# Hindi Abstractive Text Summarization — Project Report

## 1. Introduction

News consumption in Indian languages is growing faster than the NLP tooling built to
support it. Most publicly available summarization systems and benchmarks are
English-first: pretrained models are larger and better for English, and evaluation
resources (datasets, leaderboards) are comparatively scarce for Hindi despite it being
one of the most widely spoken languages in the world. This project builds an
end-to-end abstractive summarization pipeline for Hindi news articles — from raw text
to a fine-tuned transformer that produces a short summary — and compares it against
standard extractive baselines.

Concretely, given a Hindi news article, the goal is to produce a short, fluent,
faithful summary of the article's content. This project treats it as a supervised
sequence-to-sequence learning problem: fine-tune a pretrained Indic-language seq2seq
model on (article, summary) pairs, and evaluate against held-out human-written
summaries using ROUGE and BERTScore.

## 2. Related Work

**Extractive summarization.** Early summarization systems selected and concatenated
existing sentences from the source document rather than generating new text.
TextRank (Mihalcea & Tarau, 2004) and LexRank (Erkan & Radev, 2004) both apply
graph-based ranking (a PageRank variant) over sentence similarity graphs to identify
the most "central" sentences in a document — this project's TextRank baseline follows
that approach directly, using TF-IDF cosine similarity as the edge weight.

**Neural abstractive summarization.** The shift to abstractive (generate, don't just
select) summarization was driven by sequence-to-sequence models with attention, and
later by pointer-generator networks (See et al., 2017), which let the decoder either
generate a novel word or copy one directly from the source — addressing the tendency
of pure seq2seq models to hallucinate facts or mishandle rare/out-of-vocabulary words.
Liu & Lapata (2019) showed pretrained encoders (BERT) could be adapted for
extractive and abstractive summarization alike (BERTSUM). Large pretrained
sequence-to-sequence models purpose-built for generation — BART (Lewis et al., 2020)
and PEGASUS (Zhang et al., 2020), which pretrains specifically with a gap-sentence
generation objective resembling summarization — became the dominant paradigm shortly
after, and are the direct ancestors of the model used in this project.

**Multilingual and Indic-language summarization.** mT5 (Xue et al., 2021) extended
T5's text-to-text framework to 101 languages, making it a common default choice for
non-English generation tasks, Hindi included. However, general multilingual models
spread their vocabulary and capacity thin across every language they cover. IndicBART
(Dabre et al., 2022) instead pretrains an mBART-style denoising seq2seq model
specifically on 11 Indic languages plus English, giving it a Devanagari-aware
vocabulary and (per the original paper) better downstream performance on Indic
generation tasks than similarly-sized multilingual alternatives — which is why it's
the model fine-tuned in this project rather than mT5. XL-Sum (Hasan et al., 2021), the
dataset used here, was itself introduced as a 45-language summarization benchmark
built from BBC-style news sources specifically to address the lack of non-English
summarization data at scale; the original paper reports mT5 fine-tuning results across
all 45 languages as a baseline for future work to build on.

**Evaluation.** ROUGE (Lin, 2004) remains the standard n-gram-overlap metric for
summarization, and is what ILSUM (the shared task this project was originally scoped
against — see `data/README.md`) and most prior summarization work report. ROUGE has a
well-known blind spot, though: it penalizes valid paraphrases that don't share
surface n-grams with the reference. BERTScore (Zhang et al., 2020) addresses this by
comparing contextual embeddings instead of raw n-grams, and is reported here alongside
ROUGE for that reason. Separately, Maynez et al. (2020) documented how often
abstractive summarization models produce content unsupported by the source
("hallucination") even when their ROUGE scores look good — a limitation worth keeping
in mind when reading this project's own results (see `docs/ethics.md`).

## 3. Methodology

Two families of approach are implemented and compared:

1. **Extractive baselines** (no training required):
   - **Lead-N** — simply take the first N sentences of the article. A surprisingly
     strong baseline for news text, since journalistic writing conventionally front-
     loads the most important information ("inverted pyramid" style).
   - **TextRank** — build a sentence graph weighted by TF-IDF cosine similarity, run
     PageRank over it, and take the top-N ranked sentences (in their original order).

2. **Abstractive model** — fine-tune `ai4bharat/IndicBART`, an mBART-style
   pretrained Indic seq2seq model, on (article, summary) pairs. The source article is
   suffixed with a language tag (`</s> <2hi>`) as IndicBART expects, and the decoder's
   start token is set to the same tag so generation begins in Hindi (see
   `src/textsumm/model.py`). Fine-tuning uses Hugging Face's `Seq2SeqTrainer` with
   teacher forcing and cross-entropy loss; generation at inference/eval time uses beam
   search.

Sentence splitting for the extractive baselines uses `indic-nlp-library`'s Hindi
sentence tokenizer (falling back to a danda/punctuation regex if that's unavailable),
since naive period-splitting doesn't handle the Devanagari `।` sentence-ending
character.

## 4. Dataset & Experimental Setup

See `data/README.md` for full dataset details and licensing. In short: XL-Sum Hindi
(Hasan et al., 2021), ~70.7k/8.8k/8.8k train/validation/test BBC Hindi news
articles paired with a single-sentence summary each.

**Compute setup.** This was trained and evaluated on a CPU-only laptop, not a GPU
server, which directly shaped the experimental scale:

- Training subset: `<TRAIN_SAMPLES>` examples (out of ~70.7k available), `<EPOCHS>`
  epoch(s), batch size `<BATCH_SIZE>`
- Max source / target length: `<MAX_SRC>` / `<MAX_TGT>` tokens
- Evaluation subset: `<EVAL_SAMPLES>` test examples, scored with ROUGE-1/2/L F1 and
  BERTScore F1

The training and evaluation scripts (`scripts/download_data.py`, `src/textsumm/train.py`,
`scripts/run_baseline.py`, `scripts/run_eval.py`) all accept `--max-*-samples` /
`--epochs` flags precisely so this can be re-run at full dataset scale on a GPU —
this project reports the modest-scale numbers it could actually produce locally,
rather than claiming numbers from a run that didn't happen.

## 5. Results

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---|---|---|---|---|
| Lead-2 | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` |
| TextRank | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` |
| IndicBART (fine-tuned) | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` |

Raw score files: `results/baseline_scores.json`, `results/model_scores.json`.
Sample generated summaries: `results/sample_predictions.jsonl`.

## 6. Discussion

`<TODO: fill in after real results are in — e.g. does the fine-tuned model actually
beat the extractive baselines at this scale, or does the small training subset put it
behind Lead-N initially; what do the qualitative sample predictions look like;
any obvious failure modes observed (truncation, repetition, off-topic generation).>`

## 7. Conclusion & Future Work

This project implements a full pipeline — data acquisition and cleaning, two
extractive baselines, and a fine-tuned abstractive transformer — for Hindi news
summarization, evaluated with the same metrics (ROUGE, BERTScore) used by the ILSUM
shared task this project was originally scoped against.

Natural next steps, most of which just require more compute rather than a different
approach:

- Fine-tune on the full ~70.7k training set on a GPU instead of a CPU-limited subset.
- Add a lightweight factual-consistency check (e.g. an NLI-based faithfulness score)
  given the hallucination risk noted in `docs/ethics.md`.
- Extend to the other ILSUM/XL-Sum Indian languages (Gujarati, Bengali) the same
  pipeline already supports via `DATASET_CONFIG`.
- Try a pointer-generator-style copy mechanism or constrained decoding to reduce
  hallucinated entities/numbers, which is a known weak point for news summarization.

## References

- Mihalcea, R. & Tarau, P. (2004). *TextRank: Bringing Order into Text*. EMNLP.
- Erkan, G. & Radev, D. (2004). *LexRank: Graph-based Lexical Centrality as Salience in Text Summarization*. JAIR.
- See, A., Liu, P. J., & Manning, C. D. (2017). *Get To The Point: Summarization with Pointer-Generator Networks*. ACL.
- Liu, Y. & Lapata, M. (2019). *Text Summarization with Pretrained Encoders*. EMNLP-IJCNLP.
- Lewis, M. et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*. ACL.
- Zhang, J., Zhao, Y., Saleh, M., & Liu, P. J. (2020). *PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization*. ICML.
- Xue, L. et al. (2021). *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer*. NAACL.
- Dabre, R. et al. (2022). *IndicBART: A Pre-trained Model for Indic Natural Language Generation*. ACL Findings.
- Hasan, T. et al. (2021). *XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages*. ACL-IJCNLP Findings.
- Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*. ACL Workshop.
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). *BERTScore: Evaluating Text Generation with BERT*. ICLR.
- Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). *On Faithfulness and Factuality in Abstractive Summarization*. ACL.
