"""Fine-tune IndicBART on XL-Sum Hindi for abstractive summarization.

Defaults are deliberately modest (small subset, short sequences, a couple of
epochs) so the whole pipeline finishes on a laptop CPU in a reasonable time.
Bump --max-train-samples, --max-source-length, and run on a GPU box for a
serious training run - see README for the scaled-up config.
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from textsumm.data import load_processed
from textsumm.model import (
    configure_decoder_start,
    format_source,
    format_target,
    load_model,
    load_tokenizer,
)
from textsumm.utils import set_seed


def build_dataset(split: str, max_samples: int | None = None) -> Dataset:
    examples = load_processed(split)
    if max_samples is not None:
        examples = examples[:max_samples]
    return Dataset.from_list(examples)


def make_preprocess_fn(tokenizer, max_source_length: int, max_target_length: int):
    def preprocess(batch):
        sources = [format_source(t) for t in batch["text"]]
        targets = [format_target(s) for s in batch["summary"]]

        model_inputs = tokenizer(
            sources,
            max_length=max_source_length,
            truncation=True,
            add_special_tokens=False,
        )
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            add_special_tokens=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-eval-samples", type=int, default=200)
    parser.add_argument("--max-source-length", type=int, default=384)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-dir", type=str, default="checkpoints/indicbart-hindi")
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = load_tokenizer()
    model = load_model()
    configure_decoder_start(model, tokenizer)

    train_ds = build_dataset("train", args.max_train_samples)
    eval_ds = build_dataset("validation", args.max_eval_samples)

    preprocess = make_preprocess_fn(tokenizer, args.max_source_length, args.max_target_length)
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        predict_with_generate=True,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_path = Path(args.output_dir) / "train_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    print(f"training metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
