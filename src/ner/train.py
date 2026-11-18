"""Fine-tune distilbert-base-cased for token classification.

Designed to run on Google Colab / Kaggle with GPU or locally.
Inputs:
    data/ner/train.jsonl (2100 sentences)
    data/ner/val.jsonl   (450 sentences)
Output:
    models/distilbert-ner/ (weights + tokenizer)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 13 BIO Labels
LABEL_LIST = [
    "O",
    "B-PERSON", "I-PERSON",
    "B-EMAIL",  "I-EMAIL",
    "B-PHONE",  "I-PHONE",
    "B-EMP_ID", "I-EMP_ID",
    "B-ADDRESS","I-ADDRESS",
    "B-SALARY", "I-SALARY",
]
LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL: dict[int, str] = {i: lbl for i, lbl in enumerate(LABEL_LIST)}


def load_ner_jsonl(path: Path | str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def tokenize_and_align_labels(
    records: list[dict],
    tokenizer,
    max_length: int = 256,
) -> dict:
    """Tokenize words and align token-level BIO tags to subwords."""
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for r in records:
        words = r["tokens"]
        tags = r["ner_tags"]

        tokenized = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        word_ids = tokenized.word_ids()
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special token ([CLS], [SEP], [PAD])
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword token of this word
                tag = tags[word_idx] if word_idx < len(tags) else "O"
                label_ids.append(LABEL2ID.get(tag, 0))
            else:
                # Subsequent subwords get -100 to evaluate on first subword only
                label_ids.append(-100)
            previous_word_idx = word_idx

        all_input_ids.append(tokenized["input_ids"])
        all_attention_mask.append(tokenized["attention_mask"])
        all_labels.append(label_ids)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def compute_metrics(p):
    """Compute entity-level metrics."""
    import numpy as np
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)

    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]

    # Calculate micro F1
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred_seq, true_seq in zip(true_predictions, true_labels):
        pred_set = set()
        true_set = set()

        def _get_entities(seq):
            ents = []
            curr_ent = None
            for idx, tag in enumerate(seq):
                if tag.startswith("B-"):
                    if curr_ent:
                        ents.append(curr_ent)
                    curr_ent = (tag[2:], idx, idx + 1)
                elif tag.startswith("I-"):
                    if curr_ent and curr_ent[0] == tag[2:]:
                        curr_ent = (curr_ent[0], curr_ent[1], idx + 1)
                    else:
                        if curr_ent:
                            ents.append(curr_ent)
                        curr_ent = (tag[2:], idx, idx + 1)
                else:
                    if curr_ent:
                        ents.append(curr_ent)
                        curr_ent = None
            if curr_ent:
                ents.append(curr_ent)
            return set(ents)

        pred_set = _get_entities(pred_seq)
        true_set = _get_entities(true_seq)

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"precision": prec, "recall": rec, "f1": f1}


def train(
    model_name: str = "distilbert-base-cased",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    seed: int = 42,
    output_dir: str = "models/distilbert-ner",
) -> None:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        set_seed,
    )

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    train_path = PROJECT_ROOT / "data" / "ner" / "train.jsonl"
    val_path   = PROJECT_ROOT / "data" / "ner" / "val.jsonl"

    print("Loading data...")
    train_records = load_ner_jsonl(train_path)
    val_records   = load_ner_jsonl(val_path)
    print(f"Train samples: {len(train_records)}, Val samples: {len(val_records)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_enc = tokenize_and_align_labels(train_records, tokenizer)
    val_enc   = tokenize_and_align_labels(val_records, tokenizer)

    class NERDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.input_ids = encodings["input_ids"]
            self.attention_mask = encodings["attention_mask"]
            self.labels = encodings["labels"]

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
                "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    train_dataset = NERDataset(train_enc)
    val_dataset   = NERDataset(val_enc)

    print("Initializing model...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    out_path = PROJECT_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_path / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving fine-tuned model and tokenizer to {out_path}...")
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="models/distilbert-ner")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
    )
