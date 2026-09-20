"""ROUGE and BERTScore computation, matching the metrics used by ILSUM Task 1."""

from bert_score import score as bert_score
from rouge_score import rouge_scorer


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = len(predictions)
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
    return {key: value / n for key, value in totals.items()} if n else totals


def compute_bertscore(predictions: list[str], references: list[str], lang: str = "hi") -> dict:
    precision, recall, f1 = bert_score(predictions, references, lang=lang, verbose=False)
    return {
        "bertscore_precision": precision.mean().item(),
        "bertscore_recall": recall.mean().item(),
        "bertscore_f1": f1.mean().item(),
    }


def evaluate_predictions(predictions: list[str], references: list[str]) -> dict:
    results = compute_rouge(predictions, references)
    results.update(compute_bertscore(predictions, references))
    return results
