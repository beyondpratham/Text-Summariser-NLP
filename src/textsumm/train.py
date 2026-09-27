"""Fine-tune IndicBART on XL-Sum Hindi for abstractive summarization.

Laptop smoke test (CPU, a few minutes per 100 steps):
    python -m textsumm.train --max-train-samples 300 --epochs 1 --eval-steps 50

Full run (single GPU, e.g. a Kaggle/Colab T4):
    python -m textsumm.train --max-train-samples 0 --epochs 3 --batch-size 16 --grad-accum 2

The model is evaluated with ROUGE on a validation subset during training and
the best checkpoint by ROUGE-L is kept, with early stopping if it plateaus.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from textsumm.data import load_processed
from textsumm.evaluate import compute_rouge
from textsumm.model import MODEL_NAME, clean_output, configure_generation, encode, load_model, load_tokenizer
from textsumm.utils import set_seed


def make_preprocess_fn(tokenizer, max_source_length: int, max_target_length: int):
    def preprocess(batch):
        input_ids = [encode(tokenizer, text, max_source_length) for text in batch["text"]]
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
            "labels": [encode(tokenizer, summary, max_target_length) for summary in batch["summary"]],
        }

    return preprocess


def make_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = [clean_output(t) for t in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
        decoded_labels = [clean_output(t) for t in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        metrics = compute_rouge(decoded_preds, decoded_labels)
        metrics["gen_words"] = float(np.mean([len(p.split()) for p in decoded_preds]))
        return metrics

    return compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="checkpoint to start from")
    parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="keep the shared 64k-token embedding matrix fixed (saves memory on small machines)",
    )
    parser.add_argument("--max-train-samples", type=int, default=2000, help="0 = use the full training set")
    parser.add_argument(
        "--train-offset", type=int, default=0, help="skip this many training examples (to continue on unseen data)"
    )
    parser.add_argument("--max-eval-samples", type=int, default=200)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--num-beams", type=int, default=4, help="beam size for validation generation")
    parser.add_argument("--output-dir", type=str, default="checkpoints/indicbart-hindi")
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    import torch
    from datasets import Dataset
    from transformers import (
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    tokenizer = load_tokenizer(args.model_name)
    model = configure_generation(load_model(args.model_name), tokenizer)
    model.generation_config.no_repeat_ngram_size = 3
    if args.freeze_embeddings:
        model.get_input_embeddings().weight.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable / 1e6:.1f}M")

    limit = args.train_offset + args.max_train_samples if args.max_train_samples else None
    train_examples = load_processed("train", limit)[args.train_offset :]
    eval_examples = load_processed("validation", args.max_eval_samples or None)
    preprocess = make_preprocess_fn(tokenizer, args.max_source_length, args.max_target_length)
    train_ds = Dataset.from_list(train_examples).map(preprocess, batched=True, remove_columns=["id", "text", "summary"])
    eval_ds = Dataset.from_list(eval_examples).map(preprocess, batched=True, remove_columns=["id", "text", "summary"])

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    eval_steps = min(args.eval_steps, total_steps)
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=int(total_steps * args.warmup_ratio),
        label_smoothing_factor=args.label_smoothing,
        num_train_epochs=args.epochs,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.num_beams,
        seed=args.seed,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    configure_generation(trainer.model, tokenizer)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "args": vars(args),
        "train_examples": len(train_ds),
        "total_steps": total_steps,
        "train": train_result.metrics,
        "eval": eval_metrics,
    }
    out_path = Path(args.output_dir) / "run_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"best checkpoint saved to {args.output_dir}; run summary in {out_path}")


if __name__ == "__main__":
    main()
