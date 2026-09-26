import numpy as np
import pytest

from textsumm.evaluate import (
    bootstrap_ci,
    compute_rouge,
    evaluate_predictions,
    has_unsupported_number,
    novel_ngram_ratio,
)


def test_compute_rouge_perfect_match():
    scores = compute_rouge(["दिल्ली में भारी बारिश हुई।"], ["दिल्ली में भारी बारिश हुई।"])
    assert scores["rouge1"] == 1.0
    assert scores["rougeL"] == 1.0


def test_compute_rouge_ignores_punctuation():
    scores = compute_rouge(["दिल्ली में भारी बारिश हुई"], ["दिल्ली में, भारी बारिश हुई।"])
    assert scores["rouge1"] == 1.0
    assert scores["rouge2"] == 1.0


def test_compute_rouge_no_overlap():
    scores = compute_rouge(["आसमान नीला"], ["बिल्ली सो रही"])
    assert scores["rouge1"] == 0.0


def test_compute_rouge_empty_input():
    assert compute_rouge([], []) == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def test_compute_rouge_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_rouge(["एक"], [])


def test_bootstrap_ci_brackets_the_mean():
    values = np.random.default_rng(1).random(500)
    low, high = bootstrap_ci(values)
    assert low < values.mean() < high
    assert high - low < 0.1


def test_novel_ngram_ratio():
    source = "सरकार ने नई नीति की घोषणा की"
    assert novel_ngram_ratio("सरकार ने नई नीति", source, n=2) == 0.0
    assert novel_ngram_ratio("विपक्ष ने विरोध किया", source, n=1) == pytest.approx(3 / 4)


def test_unsupported_number_detection_handles_devanagari_digits():
    source = "हादसे में २५ लोगों की मौत हुई"
    assert not has_unsupported_number("हादसे में 25 लोगों की मौत", source)
    assert has_unsupported_number("हादसे में 52 लोगों की मौत", source)


def test_evaluate_predictions_reports_cis_and_diagnostics():
    preds = ["दिल्ली में बारिश हुई", "बाढ़ से 30 घर डूबे"]
    refs = ["दिल्ली में भारी बारिश", "बाढ़ से 20 घर डूबे"]
    sources = ["दिल्ली में आज भारी बारिश हुई", "बाढ़ से 20 घर डूब गए"]
    results = evaluate_predictions(preds, refs, sources, bertscore=False, n_resamples=100)
    assert results["num_examples"] == 2
    low, high = results["rouge1_ci95"]
    assert low <= results["rouge1"] <= high
    assert results["unsupported_number_rate"] == 0.5
    assert "bertscore_f1" not in results
