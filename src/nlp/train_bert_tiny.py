import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "prajjwal1/bert-tiny"


def compute_metrics(eval_pred):
    import numpy as np

    logits, labels = eval_pred
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits[:, 1] / np.sum(exp_logits, axis=1)
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
    }


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["text", "label"]].dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT-tiny for phishing email detection")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="artifacts/bert_tiny")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_df = load_df(data_dir / "train.csv")
    val_df = load_df(data_dir / "val.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True)

    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in {"input_ids", "attention_mask", "label"}])
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in {"input_ids", "attention_mask", "label"}])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved BERT-tiny artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
