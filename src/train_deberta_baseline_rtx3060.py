#!/usr/bin/env python3
# src/train_deberta_weighted.py
"""
Train DeBERTa for EDOS Task A with surgical fixes:
 - class-weighted loss (WeightedTrainer)
 - optional upsampling of minority class
 - OOM mitigations (small batch, gradient accumulation, bf16/fp16)
 - single, clear classification report per evaluation
 - timing for training and evaluation

Usage:
  python src/train_deberta_weighted.py --model_name_or_path microsoft/deberta-v3-large
"""

from pathlib import Path
import argparse
import logging
from datetime import datetime
import math
import os
import sys
import time

import pandas as pd
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import Dataset, DatasetDict
import evaluate

# -------------------------
# Weighted Trainer
# -------------------------
class WeightedTrainer(Trainer):
    """
    Trainer that applies class weights to CrossEntropyLoss.
    Pass `class_weights` as a torch.tensor on CPU; it will be moved to device.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# -------------------------
# Utilities
# -------------------------
def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)

def upsample_minority(df: pd.DataFrame, label_col="label", random_state=42):
    """Simple upsampling of minority class to match majority count."""
    counts = df[label_col].value_counts()
    if len(counts) <= 1:
        return df
    majority_label = counts.idxmax()
    majority_count = counts.max()
    parts = []
    for lbl, grp in df.groupby(label_col):
        if lbl == majority_label:
            parts.append(grp)
        else:
            n_repeat = math.ceil(majority_count / len(grp))
            replicated = pd.concat([grp] * n_repeat, ignore_index=True)
            replicated = replicated.sample(n=majority_count, random_state=random_state).reset_index(drop=True)
            parts.append(replicated)
    upsampled = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return upsampled

def compute_and_log_class_weights(labels: np.ndarray, logger):
    classes = np.unique(labels)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    cw_map = {int(c): float(w) for c, w in zip(classes, cw)}
    logger.info(f"Computed class weights: {cw_map}")
    max_label = int(classes.max())
    weights = [cw_map.get(i, 1.0) for i in range(max_label + 1)]
    return torch.tensor(weights, dtype=torch.float)

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train DeBERTa with class-weighted loss and OOM mitigations")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_csv", default="data/processed/task_a_train.csv")
    parser.add_argument("--dev_csv",   default="data/processed/task_a_dev.csv")
    parser.add_argument("--test_csv",  default="data/processed/task_a_test.csv")
    parser.add_argument("--output_dir", default="models/deberta/task_a_weighted")
    parser.add_argument("--seed", type=int, default=42)

    # memory / performance
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Per-device batch size. Lower this if you hit OOM.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Accumulate gradients to achieve larger effective batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=256)

    # training schedule
    parser.add_argument("--learning_rate", type=float, default=1.2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # class imbalance options
    parser.add_argument("--use_class_weights", action="store_true", default=True,
                        help="Compute class weights and use them in the loss (recommended).")
    parser.add_argument("--upsample_minority", action="store_true", default=False,
                        help="Upsample minority class in the training CSV before tokenization.")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bf16 if available. If not, set --fp16 to use fp16.")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use fp16 mixed precision (fallback).")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # logging
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"train_deberta_weighted_{timestamp}.log"
    logger = setup_logger(log_file)
    logger.info(f"Logging to: {log_file}")
    logger.info(f"Arguments: {args}")

    set_seed(args.seed)

    # Read CSVs (optionally upsample training CSV)
    logger.info("Loading CSVs...")
    train_df = pd.read_csv(args.train_csv, dtype=str, keep_default_na=False)
    dev_df = pd.read_csv(args.dev_csv, dtype=str, keep_default_na=False)
    test_df = pd.read_csv(args.test_csv, dtype=str, keep_default_na=False)
    for df in (train_df, dev_df, test_df):
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV files must contain 'text' and 'label' columns.")
        df["label"] = df["label"].astype(int)

    logger.info(f"Train counts before upsampling: {train_df['label'].value_counts().to_dict()}")
    if args.upsample_minority:
        logger.info("Upsampling minority class in training set...")
        train_df = upsample_minority(train_df, label_col="label", random_state=args.seed)
        logger.info(f"Train counts after upsampling: {train_df['label'].value_counts().to_dict()}")

    # Convert to HuggingFace Datasets
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    dev_ds = Dataset.from_pandas(dev_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))
    dataset_dict = DatasetDict({"train": train_ds, "validation": dev_ds, "test": test_ds})

    # Tokenizer and model
    logger.info(f"Loading tokenizer and model from: {args.model_name_or_path}")
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    # Tokenization function
    def preprocess_function(examples):
        texts = examples["text"]
        return tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=args.max_seq_length,
        )

    logger.info("Tokenizing datasets (this may take a while)...")
    tokenized = dataset_dict.map(preprocess_function, batched=True, remove_columns=dataset_dict["train"].column_names)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # Compute class weights if requested
    class_weights_tensor = None
    if args.use_class_weights:
        labels = np.array(train_df["label"].astype(int).values)
        class_weights_tensor = compute_and_log_class_weights(labels, logger)
        logger.info(f"Class weights tensor: {class_weights_tensor.tolist()}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        save_total_limit=3,
        seed=args.seed,
        dataloader_num_workers=4,
        report_to="none",
    )

    # Metrics: basic metrics used by Trainer (we will compute classification_report separately)
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
        f1_macro = metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
        n_pred_sexist = int((preds == 1).sum())
        n_pred_not = int((preds == 0).sum())
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "n_pred_sexist": n_pred_sexist,
            "n_pred_not_sexist": n_pred_not,
        }

    # Instantiate WeightedTrainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
    )

    # Train with timing
    try:
        logger.info("Starting training...")
        t0 = time.time()
        train_result = trainer.train()
        t1 = time.time()
        train_time = t1 - t0
        logger.info(f"Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes).")
    except RuntimeError as e:
        msg = str(e).lower()
        logger.exception("Training failed with RuntimeError.")
        if "out of memory" in msg or ("cuda" in msg and "out of memory" in msg):
            logger.error("CUDA OOM detected during training. Suggested actions:")
            logger.error(" - Reduce --per_device_train_batch_size (e.g., to 4 or 2).")
            logger.error(" - Increase --gradient_accumulation_steps to keep effective batch size.")
            logger.error(" - Use --bf16 (if supported) or --fp16 to reduce memory.")
            logger.error(" - Use a smaller model (e.g., deberta-v3-base or -small).")
            logger.error(" - Reduce --max_seq_length (e.g., 128).")
        raise

    # Evaluate on validation with timing and single classification report
    logger.info("Evaluating on validation set (will compute single classification report)...")
    t0 = time.time()
    val_pred_out = trainer.predict(tokenized["validation"])
    t1 = time.time()
    val_time = t1 - t0
    logger.info(f"Validation prediction completed in {val_time:.2f} seconds ({val_time/60:.2f} minutes).")

    val_logits = val_pred_out.predictions
    val_preds = np.argmax(val_logits, axis=-1)
    val_labels = val_pred_out.label_ids

    # Basic metrics (as Trainer would show)
    val_metrics = compute_metrics((val_logits, val_labels))
    logger.info(f"Validation basic metrics: {val_metrics}")

    # Single, clear classification report for validation
    logger.info("Validation Classification report (computed from the validation dataset):")
    val_clf_report = classification_report(val_labels, val_preds, target_names=["not_sexist", "sexist"], digits=4, zero_division=0)
    logger.info("\n" + val_clf_report)
    logger.info(f"Validation confusion matrix:\n{confusion_matrix(val_labels, val_preds)}")

    # Evaluate on test with timing and single classification report
    logger.info("Evaluating on test set (will compute single classification report)...")
    t0 = time.time()
    test_pred_out = trainer.predict(tokenized["test"])
    t1 = time.time()
    test_time = t1 - t0
    logger.info(f"Test prediction completed in {test_time:.2f} seconds ({test_time/60:.2f} minutes).")

    test_logits = test_pred_out.predictions
    test_preds = np.argmax(test_logits, axis=-1)
    test_labels = test_pred_out.label_ids

    test_metrics = compute_metrics((test_logits, test_labels))
    logger.info(f"Test basic metrics: {test_metrics}")

    # Single, clear classification report for test
    logger.info("Test Classification report (computed from the test dataset):")
    test_clf_report = classification_report(test_labels, test_preds, target_names=["not_sexist", "sexist"], digits=4, zero_division=0)
    logger.info("\n" + test_clf_report)
    logger.info(f"Test confusion matrix:\n{confusion_matrix(test_labels, test_preds)}")

    # Log summary of timings
    logger.info("=== Timing summary ===")
    logger.info(f"Training time (wall-clock): {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    logger.info(f"Validation prediction time : {val_time:.2f} seconds ({val_time/60:.2f} minutes)")
    logger.info(f"Test prediction time       : {test_time:.2f} seconds ({test_time/60:.2f} minutes)")

    # Save final model and tokenizer
    logger.info("Saving final model and tokenizer...")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"Saved artifacts to: {out_dir.resolve()}")

    # Save test predictions CSV with preds and probs
    logger.info("Generating and saving test predictions CSV...")
    probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    test_df_out = test_df.copy().reset_index(drop=True)
    test_df_out["pred"] = test_preds
    test_df_out["prob_not_sexist"] = probs[:, 0]
    test_df_out["prob_sexist"] = probs[:, 1]
    preds_csv = out_dir / f"predictions_test_{timestamp}.csv"
    test_df_out.to_csv(preds_csv, index=False)
    logger.info(f"Saved test predictions to: {preds_csv}")

if __name__ == "__main__":
    main()
