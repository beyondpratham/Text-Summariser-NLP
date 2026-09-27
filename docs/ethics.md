# Ethical Considerations & Social Impact

## Why this matters

Hindi has ~600M speakers, yet the vast majority of NLP tooling — summarization
included — is built and evaluated almost exclusively on English. Most "multilingual"
summarization benchmarks under-represent Indian languages both in dataset size and in
how well pretrained models actually perform on them, compared to high-resource
languages. Building and open-sourcing a working Hindi summarizer, even a modest one,
is a small step toward closing that gap — it lowers the bar for a Hindi-language news
aggregator, a regional-language accessibility tool, or a student project to reuse
working code instead of starting from nothing.

## Risks and limitations

- **Hallucination.** Abstractive models can generate summaries that are fluent but
  factually inconsistent with the source article (a well-documented failure mode in
  summarization literature, e.g. Maynez et al., 2020). This model is not fact-checked
  against its source text at inference time, so it should not be used to produce
  summaries presented as ground truth without human review — particularly not for
  news, medical, or legal content where a fabricated detail could cause real harm.
  The evaluation reports an *unsupported-number rate* (summaries containing a
  number that never appears in the article) as a rough measure of this. It
  catches invented figures only, not wrong names, places, or claims.
  For the released model, 3.1% of test summaries contain such a number, and
  manual review found wrong facts the metric misses (for example, a city placed
  in the wrong state). See section 6 of the report.
- **Domain and register narrowness.** The model is fine-tuned on BBC Hindi news
  articles. It will likely perform worse on other registers (conversational text,
  regional dialects, technical or legal documents) and shouldn't be assumed to
  generalize beyond news-style writing.
- **Training data bias.** BBC News editorial choices (what's newsworthy, how events
  are framed) are baked into the training data and, by extension, into what the model
  learns to consider "summary-worthy." This is worth keeping in mind before using the
  model on content far outside the BBC's editorial scope.
- **Compute-scale caveat.** The reported model was fine-tuned on a laptop CPU on
  1,600 of the 70,777 training articles, using only the first 256 tokens of each
  article, not the full dataset on a GPU. Results here should be read as a proof of concept,
  not a state-of-the-art claim; see the README for how to scale the training run up.

## Intended use

This is a personal/educational project meant to demonstrate an end-to-end NLP
pipeline (data cleaning, extractive baselines, transformer fine-tuning, evaluation).
It is not intended for production deployment, and definitely not for any
use case where a wrong or fabricated summary could mislead someone in a
consequential decision.
