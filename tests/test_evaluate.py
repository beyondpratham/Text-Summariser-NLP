from textsumm.evaluate import compute_rouge


def test_compute_rouge_perfect_match():
    preds = ["दिल्ली में भारी बारिश हुई।"]
    refs = ["दिल्ली में भारी बारिश हुई।"]
    scores = compute_rouge(preds, refs)
    assert scores["rouge1"] == 1.0
    assert scores["rougeL"] == 1.0


def test_compute_rouge_no_overlap():
    preds = ["आसमान नीला है।"]
    refs = ["बिल्ली सो रही है।"]
    scores = compute_rouge(preds, refs)
    assert scores["rouge1"] < 0.5


def test_compute_rouge_empty_input():
    assert compute_rouge([], []) == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
