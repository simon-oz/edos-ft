#!/usr/bin/env python3
"""
train_deberta_baseline.py
Train DeBERTa-v3-large on EDOS Task A (binary sexism detection).

This establishes the baseline F1 score and produces a trained model
for subsequent error mining and ensemble use.

Input:
  data/processed/task_a_{train,dev,test}.csv

Output:
  models/deberta/task_a_baseline/          — checkpoints and final model
  logs/train_deberta_baseline_*.log        — training log

Usage:
  From project root:  python src/train_deberta_baseline.py
  From src/ folder:   python train_deberta_baseline.py
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, classification_report
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

# ------------------------------------------------------------------
# Auto-detect project root
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "deberta" / "task_a_baseline"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Logging setup: console + file
# ------------------------------------------------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_deberta_baseline_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")
logger.info(f"Project root: {PROJECT_ROOT.absolute()}")
logger.info(f"Model output: {MODEL_DIR.absolute()}")

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 256
SEED = 42

set_seed(SEED)

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Loading Task A datasets")
logger.info("=" * 60)

train_path = PROC_DIR / "task_a_train.csv"
dev_path   = PROC_DIR / "task_a_dev.csv"
test_path  = PROC_DIR / "task_a_test.csv"

for p in [train_path, dev_path, test_path]:
    if not p.exists():
        logger.error(f"Missing: {p}. Run parse_edos_data.py first.")
        sys.exit(1)

dataset = load_dataset(
    "csv",
    data_files={
        "train": str(train_path),
        "validation": str(dev_path),
        "test": str(test_path),
    },
)

logger.info(f"Train      : {len(dataset['train'])} samples")
logger.info(f"Validation : {len(dataset['validation'])} samples")
logger.info(f"Test       : {len(dataset['test'])} samples")

# Class distribution
labels_train = dataset["train"]["label"]
logger.info(f"Train class distribution: sexist={sum(labels_train)}, "
            f"not_sexist={len(labels_train) - sum(labels_train)}")

# ------------------------------------------------------------------
# 2. Tokenize
# ------------------------------------------------------------------
logger.info("=" * 60)
logger.info(f"Loading tokenizer: {MODEL_NAME}")
logger.info("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ------------------------------------------------------------------
# 3. Load model
# ------------------------------------------------------------------
logger.info("=" * 60)
logger.info(f"Loading model: {MODEL_NAME}")
logger.info("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=False,
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Total parameters     : {total_params:,}")
logger.info(f"Trainable parameters : {trainable_params:,}")

# ------------------------------------------------------------------
# 4. Metrics
# ------------------------------------------------------------------
def compute_metrics(eval_pred):
    """Compute macro F1, precision, recall for binary classification."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = (preds == labels).mean()

    # Per-class F1 for detailed logging
    per_class = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_not_sexist": per_class[2][0] if len(per_class[2]) > 0 else 0.0,
        "f1_sexist": per_class[2][1] if len(per_class[2]) > 1 else 0.0,
    }

# ------------------------------------------------------------------
# 5. Training arguments
# ------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Configuring training")
logger.info("=" * 60)

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    warmup_ratio=0.1,
    logging_steps=50,
    logging_dir=str(PROJECT_ROOT / "logs" / "tensorboard" / "deberta_task_a"),
    report_to=["tensorboard"],  # Disable wandb by default; enable if configured
    seed=SEED,
    fp16=True,
    dataloader_num_workers=4,
    remove_unused_columns=True,
)

logger.info(f"Learning rate        : {training_args.learning_rate}")
logger.info(f"Train batch size     : {training_args.per_device_train_batch_size}")
logger.info(f"Eval batch size      : {training_args.per_device_eval_batch_size}")
logger.info(f"Max epochs           : {training_args.num_train_epochs}")
logger.info(f"Weight decay         : {training_args.weight_decay}")
logger.info(f"Warmup ratio         : {training_args.warmup_ratio}")
logger.info(f"Early stopping       : patience=3, metric=f1_macro")

# ------------------------------------------------------------------
# 6. Trainer
# ------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ------------------------------------------------------------------
# 7. Train
# ------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Starting training")
logger.info("=" * 60)

train_result = trainer.train()

logger.info("Training complete.")
logger.info(f"  Final train loss : {train_result.training_loss:.4f}")
logger.info(f"  Best checkpoint  : {trainer.state.best_model_checkpoint}")
logger.info(f"  Best F1 (dev)    : {trainer.state.best_metric:.4f}")

# Save training metrics
train_metrics = train_result.metrics
trainer.save_metrics("train", train_metrics)

# ------------------------------------------------------------------
# 8. Evaluate on test set
# ------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Evaluating on TEST set")
logger.info("=" * 60)

test_results = trainer.evaluate(tokenized_datasets["test"])

logger.info("Test set results:")
for key, value in test_results.items():
    if isinstance(value, float):
        logger.info(f"  {key:25s} : {value:.4f}")
    else:
        logger.info(f"  {key:25s} : {value}")

# Save test metrics
trainer.save_metrics("test", test_results)

# ------------------------------------------------------------------
# 9. Save model and tokenizer
# ------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Saving final model")
logger.info("=" * 60)

trainer.save_model(MODEL_DIR / "final")
tokenizer.save_pretrained(MODEL_DIR / "final")

# Save a summary JSON
summary = {
    "model": MODEL_NAME,
    "task": "Task A - Binary Sexism Detection",
    "seed": SEED,
    "max_length": MAX_LENGTH,
    "train_samples": len(dataset["train"]),
    "dev_samples": len(dataset["validation"]),
    "test_samples": len(dataset["test"]),
    "best_checkpoint": trainer.state.best_model_checkpoint,
    "best_dev_f1": float(trainer.state.best_metric) if trainer.state.best_metric else None,
    "test_f1_macro": float(test_results.get("eval_f1_macro", 0)),
    "test_accuracy": float(test_results.get("eval_accuracy", 0)),
    "test_precision_macro": float(test_results.get("eval_precision_macro", 0)),
    "test_recall_macro": float(test_results.get("eval_recall_macro", 0)),
    "timestamp": timestamp,
}

with open(MODEL_DIR / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

logger.info(f"Summary saved to: {MODEL_DIR / 'training_summary.json'}")
logger.info("=" * 60)
logger.info("Baseline training complete!")
logger.info("=" * 60)
