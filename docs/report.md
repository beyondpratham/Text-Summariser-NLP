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

Three families of system are implemented and compared.

1. **Extractive baselines** (no training):
   - **Lead-N** takes the first N sentences. It is a strong baseline for news
     because journalists put the most important information first ("inverted pyramid").
   - **TextRank** builds a sentence graph weighted by TF-IDF cosine similarity, runs
     PageRank over it, and returns the top-N sentences in their original order. Both
     the TF-IDF weighting (smoothed idf, L2-normalized rows) and PageRank (power
     iteration, damping 0.85) are implemented directly in NumPy.
   - **Extractive oracle** greedily adds the sentence that most improves ROUGE-1 +
     ROUGE-2 against the reference, stopping when no sentence helps (Nallapati et al.,
     2017). It reads the reference, so it is an upper bound for sentence
     selection rather than a usable system.

2. **Abstractive model**: fine-tune an IndicBART checkpoint on (article, summary)
   pairs with teacher forcing and label-smoothed cross-entropy. Inference uses beam
   search (4 beams) with trigram repetition blocking.

   The starting checkpoint is `ai4bharat/MultiIndicSentenceSummarization` (called
   IndicBART-SS below): IndicBART already fine-tuned by its authors on 431K Indic
   sentence-summarization pairs. It uses the same tokenizer and sequence format as base
   IndicBART, so the pipeline runs unchanged. Two zero-shot checks on 24 validation
   articles decided the choice. Base IndicBART, pretrained only to reconstruct its
   input, mostly copied article sentences. IndicBART-SS already wrote on-topic
   abstractive summaries, just shorter than XL-Sum's. With a CPU-only compute budget,
   learning XL-Sum's length and style needs far fewer updates than learning to
   summarize from scratch. IndicBART-SS also generates 2.3x faster, because its
   outputs are shorter.

**Tokenization for Hindi.** Two standard tools fail silently on Devanagari. Python's
`\w` does not match vowel signs (matras), so scikit-learn's default TF-IDF pattern
extracts no tokens at all from an ordinary sentence like "दिल्ली में भारी बारिश हुई।".
`rouge_score` keeps only `[a-z0-9]`, so every Hindi pair scores zero. Both TextRank and
ROUGE therefore use one shared tokenizer that splits on Unicode punctuation, symbol,
and separator categories, lowercases, and maps Devanagari digits to ASCII. Sentence
splitting uses `indic-nlp-library`, which handles the danda (।) and Hindi abbreviations.

**IndicBART sequence format.** IndicBART follows mBART's format: the encoder input is
`article </s> <2hi>` and the labels are `summary </s> <2hi>`. mBART builds the decoder
input by rotating the last label token to the front, giving `<2hi> summary </s>`, so
decoding starts from `<2hi>` in both training and inference. An earlier version of
this pipeline had three bugs here, all of which failed silently:

- The `</s> <2hi>` suffix was appended to the raw text *before* truncation, so it was
  cut off every article longer than the length limit, which is nearly all of them.
- The same happened to labels, so long summaries lost their EOS token and the model
  never learned to stop. Every generated summary ran to the length cap.
- Labels were `summary </s>`, so the rotated decoder input started with `</s>` in
  training but `<2hi>` at inference.

The fixed version truncates the text and then appends the suffix. It sets the
generation special-token ids explicitly, because IndicBART's tokenizer reports
`[CLS]`/`[SEP]` as BOS/EOS, and Trainer copies those onto the model config. It also
strips leftover tags after decoding, because `<2hi>` and `</s>` are registered as
non-special tokens and survive `skip_special_tokens`.

**Evaluation.** Every system is scored with ROUGE-1/2/L F1 (averaged per example)
and a 95% percentile-bootstrap confidence interval (1,000 resamples). BERTScore is
available but optional. Two cheap diagnostics are reported alongside ROUGE: the
**novel bigram ratio** (share of summary bigrams absent from the source, where 0 means
pure copying) and the **unsupported-number rate** (share of summaries containing a
number that never appears in the source), a narrow but useful signal for hallucination.

## 4. Dataset & Experimental Setup

XL-Sum Hindi (Hasan et al., 2021) contains 70,778 / 8,847 / 8,847 train / validation /
test BBC Hindi articles, each paired with a summary written by an editor. Preprocessing
applies NFC normalization, removes zero-width characters and URLs, collapses
whitespace, and drops duplicate IDs and pairs whose summary is not shorter than the
article (one training pair was dropped). Corpus statistics
(`scripts/dataset_stats.py`, test split):

| Statistic | Value |
|---|---|
| Article length, words (mean / median / p95) | 460 / 376 / 1,021 |
| Sentences per article (mean) | 23.8 |
| Summary length, words (mean / p95) | 26.9 / 40 |
| Compression ratio (mean) | 18.1x |
| Summary unigrams not in article | 30.9% |
| Summary bigrams not in article | 73.2% |

The last row is the most important one. Most reference summaries are paraphrases,
not copied sentences, which caps what extractive methods can reach.

**Model settings.** The fine-tuned model was trained on a laptop CPU (Intel
i5-8250U, 4 cores, 8 GB RAM, no GPU), which set the scale of the experiment:

| Setting | Value |
|---|---|
| Starting checkpoint | `ai4bharat/MultiIndicSentenceSummarization` |
| Training examples | 1,600 (first 1,600 of the 70,777-example train split) |
| Max source / target tokens | 256 / 64 |
| Batch size | 4, gradient accumulation 2 (effective 8) |
| Optimizer steps | 200 (1 epoch), about 3.5 hours |
| Learning rate | 1e-4, 5% linear warmup then linear decay; AdamW, weight decay 0.01 |
| Regularization | label smoothing 0.1; shared embedding matrix frozen |
| Model selection | validation ROUGE-L on 64 articles every 50 steps; best kept |

Freezing the 64k x 1024 shared embedding matrix, which is tied to the output
layer, leaves 178.5M of the model's 244M parameters trainable. It saves about
0.8 GB of AdamW state, which is what makes training fit in 8 GB of RAM. The
256-token source limit covers roughly the first 215 words, about half the median
article. News puts the key facts first (inverted pyramid), so that part carries
most of the summary-relevant content.

Validation scores over training:

| Step | Train loss | Val loss | Val ROUGE-1 | Val ROUGE-L | Val words |
|---|---|---|---|---|---|
| 50 | 4.95 | 4.17 | 26.1 | 21.7 | 18.2 |
| 100 | 4.58 | 3.99 | 29.4 | 23.9 | 18.0 |
| 150 | 4.57 | 3.96 | 29.5 | **24.0** | 17.5 |
| 200 | 4.64 | 3.95 | 29.2 | 23.4 | 18.6 |

Step 150 was selected. Training losses include label smoothing, so they sit
above the unsmoothed validation loss.

## 5. Results

Every system is scored on the same 1,000 test articles (the first 1,000 of the
8,847-article test split), because beam-search generation on a CPU takes about 4
seconds per article. ROUGE F1 uses the Devanagari-aware tokenizer; ± is the
half-width of the 95% bootstrap confidence interval. BERTScore uses
`bert-base-multilingual-cased`. Scores ×100.

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Words | Novel bigrams | Unsupported numbers |
|---|---|---|---|---|---|---|---|
| Lead-1 | 22.15 ± 0.64 | 4.75 ± 0.37 | 16.62 ± 0.51 | 70.72 | 22.9 | 0% | 0% |
| Lead-2 | 24.67 ± 0.57 | 5.72 ± 0.35 | 17.01 ± 0.43 | 70.68 | 43.8 | 0% | 0% |
| Lead-3 | 23.77 ± 0.49 | 5.80 ± 0.32 | 15.92 ± 0.35 | 70.37 | 64.5 | 0% | 0% |
| TextRank (1 sentence) | 24.11 ± 0.58 | 4.97 ± 0.36 | 17.04 ± 0.48 | 70.53 | 30.5 | 0% | 0% |
| IndicBART-SS, zero-shot | 18.80 ± 0.64 | 4.70 ± 0.40 | 15.38 ± 0.58 | 69.62 | 10.1 | 38.4% | 1.3% |
| **IndicBART-SS, fine-tuned** | **26.76** ± 0.73 | **7.46** ± 0.53 | **21.57** ± 0.65 | **73.05** | 17.7 | 46.7% | 3.1% |
| *Extractive oracle* | *38.20* ± 0.53 | *14.63* ± 0.55 | *25.82* ± 0.54 | *74.27* | 38.1 | 1.4% | 0% |

The extractive baselines were also scored on the full test set. There they come
within 0.6 points of the subset numbers, which shows the subset is representative:

| Method (all 8,847 articles) | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Lead-1 | 22.75 ± 0.22 | 5.31 ± 0.14 | 17.23 ± 0.18 |
| Lead-2 | 25.12 ± 0.19 | 6.14 ± 0.12 | 17.31 ± 0.14 |
| Lead-3 | 24.16 ± 0.16 | 6.22 ± 0.11 | 16.10 ± 0.12 |
| TextRank (1 sentence) | 24.28 ± 0.21 | 5.23 ± 0.14 | 17.46 ± 0.17 |
| *Extractive oracle* | *38.47* ± 0.19 | *14.83* ± 0.20 | *26.09* ± 0.20 |

Raw outputs: `results/baseline_scores.json` (full test),
`results/baseline_scores_test1k.json`, `results/indicbart_zeroshot_scores.json`,
`results/indicbart_finetuned_scores.json`, and the matching `*_predictions.jsonl`
sample files.

**History.** An initial proof-of-concept run of the pipeline, before the fixes in
Section 3, trained base IndicBART on 300 examples. It reached ROUGE-1 23.0 on 200 test
articles, below Lead-N. Every one of its saved predictions began with a literal
`<2hi>` tag and ran to the 64-token cap without ending, which is how the EOS and
decoding bugs were found.

## 6. Discussion

**Lead-N is hard to beat by sentence selection.** Lead-1, Lead-2 and TextRank-1
land within 0.25 ROUGE-L points of each other, and the confidence intervals of Lead-2
and TextRank-1 overlap. BBC's inverted-pyramid style puts the key facts up front, so
position alone is a strong signal. Lead-2 has the best ROUGE-1 mainly because it is
longer (44 words vs. 27 for the average reference), which buys recall. The shorter
TextRank-1 matches it on ROUGE-L. Adding a third sentence lowers ROUGE-1 and ROUGE-L,
because the extra words cost more in precision than they add in recall.

**The ceiling for extraction is low.** The oracle, which picks sentences with the
reference in hand, reaches only 38.5 ROUGE-1. That matches the corpus statistics:
73% of reference bigrams do not appear in the article at all. Closing the gap between
roughly 25 and 38, and going past it, requires generating new text. That is the case
for the abstractive model.

**Tokenization mattered more than the algorithm.** On a 300-article sample with the
original scikit-learn tokenizer and whitespace ROUGE, TextRank scored 21.1 ROUGE-1 and
trailed Lead-2 by more than two points. With the Devanagari-aware tokenizer, TextRank
on the full test set is on par with the best Lead-N setting. The two runs differ in
sample and summary length too, so this is not a controlled comparison, but it is
consistent with the old similarity graph being nearly empty.

**Fine-tuning beats every extractive baseline, and the margin is significant.**
ROUGE-L improves by 4.5 points over the best baseline (21.57 vs. 17.04), far
outside either confidence interval. ROUGE-2 and BERTScore also improve (+1.7 and
+2.3). ROUGE-1 improves by 2.1 points even though the model's summaries are less
than half as long as Lead-2's. ROUGE-L gains the most because the model writes one
coherent sentence in the reference's structure, where extractive systems return
whatever sentence came first.

**Most of the gain comes from fine-tuning, not the starting checkpoint.** Zero-shot,
IndicBART-SS scores *below* every extractive baseline (18.8 ROUGE-1), because it
writes 10-word headlines. After 200 updates on 1,600 articles it gains 8.0 ROUGE-1
and 6.2 ROUGE-L, and its summaries grow to 18 words.

**Remaining weaknesses.**

- *Length.* Summaries average 17.7 words against 26.9 for the references, which
  costs recall. Output length stayed near 18 words throughout training (Section 4),
  well under the 64-token cap, so this is a style the model settled into, not
  truncation.
- *Faithfulness.* 3.1% of summaries contain a number not found in the article,
  against 0% for the extractive systems. Manual review of saved samples also finds
  errors the metric cannot catch. One summary places Malegaon in Madhya Pradesh
  instead of Maharashtra. Another produces the degenerate phrase "Iran and Iran".
  These are the failure modes described by Maynez et al. (2020).
- *Scale.* The model has seen 2.3% of the training data and only the first 256
  tokens of each article. The XL-Sum authors' mT5-base model, trained on the full
  multilingual corpus on GPUs, reports 38.6 ROUGE-1 for Hindi with its own ROUGE
  implementation. That number is not directly comparable to the scores here,
  because of the different tokenization, but it shows how much headroom a
  full-scale run has.

**Diagnostics.** Extractive systems have a novel bigram ratio of 0 and never
produce unsupported numbers, as expected for copying. The fine-tuned model's 46.7%
novel bigrams show it paraphrases rather than copies. Its 3.1% unsupported-number
rate is the cost of that, and it is the number to push down in future work.

## 7. Conclusion & Future Work

This project implements a full pipeline for Hindi news summarization: data
acquisition and cleaning, three extractive reference points, a fine-tuned abstractive
transformer, a statistically grounded evaluation harness, and a REST API for serving.
The most important lesson was about Hindi text specifically. Several standard
components (regex tokenizers, the reference ROUGE implementation, and mBART-style
sequence formatting under truncation) fail *silently* on Devanagari input. They
return plausible-looking numbers instead of errors. Each failure is now covered by
a unit test.

Next steps:

- Fine-tune on the full training set with 512-token inputs on a GPU. The steady
  validation gains in Section 4 suggest this is the most direct route to a
  substantially higher score.
- Tune the beam-search length penalty on validation data to close the gap
  between generated and reference summary length.
- Replace the unsupported-number check with an NLI-based faithfulness score, given
  the errors found in manual review.
- Extend to other XL-Sum Indian languages (Bengali, Gujarati, Marathi). IndicBART
  already covers them, and only the language tag and dataset archive name change.

## References

- Mihalcea, R. & Tarau, P. (2004). *TextRank: Bringing Order into Text*. EMNLP.
- Erkan, G. & Radev, D. (2004). *LexRank: Graph-based Lexical Centrality as Salience in Text Summarization*. JAIR.
- Nallapati, R., Zhai, F., & Zhou, B. (2017). *SummaRuNNer: A Recurrent Neural Network Based Sequence Model for Extractive Summarization of Documents*. AAAI.
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
