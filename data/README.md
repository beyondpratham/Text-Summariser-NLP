# Data

This project uses the Hindi subset of **XL-Sum** (Hasan et al., ACL-IJCNLP 2021 Findings),
a multilingual summarization dataset built from BBC News articles paired with the
single-sentence summary the BBC editors wrote for each one.

- Source: [`csebuetnlp/xlsum`](https://huggingface.co/datasets/csebuetnlp/xlsum) on the Hugging Face Hub
- Language used here: Hindi (`hindi` config)
- Split sizes: ~70.7k train / ~8.8k validation / ~8.8k test
- License: CC BY-NC-SA 4.0 (non-commercial research use only) — this repo's own code
  is MIT-licensed, but the dataset itself is not, so don't redistribute it commercially.

Raw and processed data files are gitignored (they're easy to regenerate and not worth
bloating the repo with). To fetch and clean the data yourself:

```bash
python scripts/download_data.py
```

This writes cleaned `train.jsonl` / `validation.jsonl` / `test.jsonl` files under
`data/processed/`, each line a `{"id", "text", "summary"}` record. Cleaning removes
stray URLs, zero-width Unicode characters picked up from web scraping, and collapses
whitespace — see `textsumm.utils.clean_text`.

## Why not the official ILSUM dataset?

The original brief for this project pointed at the ILSUM 2023 Task 1 dataset
(Hindi/Gujarati/Bengali news summarization). That dataset is password-protected
behind registration for a shared task that has since closed, so it's no longer
practically obtainable. XL-Sum Hindi covers the same problem — Hindi news
article-to-summary abstractive summarization — with a public, well-documented,
widely-cited dataset instead, which also makes this repo reproducible for anyone
cloning it.
