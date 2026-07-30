#!/usr/bin/env python3
"""
train_deberta_baseline.py
Train DeBERTa-v3-large on EDOS Task A (binary sexism detection).

This establishes the baseline F1 score and produces a trained model
for subsequent error mining and ensemble use.
"""

import sys
import json
import logging
import argparse
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
# Configuration & Utilities
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="microsoft/deberta-v3-large",
                        help="Path to pretrained model or HuggingFace model name")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    return parser.parse_args()

def compute_metrics_standard(eval_pred):
    """Standard compute_metrics using 0.5 threshold for Trainer eval epochs."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = (preds == labels).mean()
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

def compute_metrics_with_threshold(probs, labels, threshold):
    """Compute metrics dynamically based on a custom threshold."""
    preds = (probs >= threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = (preds == labels).mean()
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

def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

# ------------------------------------------------------------------
# Custom Trainer for Class-Weighted Loss
# ------------------------------------------------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ------------------------------------------------------------------
# Main Execution Guard (Crucial for dataloader_num_workers > 0)
# ------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    MODEL_NAME = args.model_path
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

    labels_train = np.array(dataset["train"]["label"])
    n_sexist = sum(labels_train)
    n_not_sexist = len(labels_train) - n_sexist
    logger.info(f"Train class distribution: sexist={n_sexist}, not_sexist={n_not_sexist}")

    # Calculate class weights inversely proportional to class frequencies
    weight_class_0 = len(labels_train) / (2 * n_not_sexist)
    weight_class_1 = len(labels_train) / (2 * n_sexist)
    class_weights = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float)
    logger.info(f"Class weights: [not_sexist={weight_class_0:.4f}, sexist={weight_class_1:.4f}]")

    # ------------------------------------------------------------------
    # 2. Tokenize
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH), 
        batched=True
    )

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
        torch_dtype=torch.float32,  # Ensure FP32 master weights for stable AMP
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters     : {total_params:,}")
    logger.info(f"Trainable parameters : {trainable_params:,}")

    # ------------------------------------------------------------------
    # 4. Training arguments
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Configuring training")
    logger.info("=" * 60)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,                           
        per_device_train_batch_size=args.batch_size,     
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,                    
        gradient_accumulation_steps=args.grad_accum,     
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        warmup_ratio=0.1,
        logging_steps=50,
        logging_dir=str(PROJECT_ROOT / "logs" / "tensorboard" / "deberta_task_a"),
        report_to=["tensorboard"],
        seed=SEED,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=True,
    )

    logger.info(f"Model path           : {MODEL_NAME}")
    logger.info(f"Learning rate        : {training_args.learning_rate}")
    logger.info(f"Train batch size     : {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"Max epochs           : {training_args.num_train_epochs}")
    logger.info(f"Weight decay         : {training_args.weight_decay}")
    logger.info(f"Warmup ratio         : {training_args.warmup_ratio}")
    logger.info(f"Early stopping       : patience=3, metric=f1_macro")

    # ------------------------------------------------------------------
    # 5. Trainer (Using WeightedTrainer)
    # ------------------------------------------------------------------
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics_standard,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)

    train_result = trainer.train()

    logger.info("Training complete.")
    logger.info(f"  Final train loss : {train_result.training_loss:.4f}")
    logger.info(f"  Best checkpoint  : {trainer.state.best_model_checkpoint}")
    logger.info(f"  Best F1 (dev)    : {trainer.state.best_metric:.4f}")

    train_metrics = train_result.metrics
    trainer.save_metrics("train", train_metrics)

    # ------------------------------------------------------------------
    # 7. Threshold Optimization on Validation Set
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Optimizing decision threshold on Validation set")
    logger.info("=" * 60)

    val_output = trainer.predict(tokenized_datasets["validation"])
    val_logits = val_output.predictions
    val_labels = val_output.label_ids
    
    # Convert logits to probabilities for the positive class (sexist = 1)
    val_probs = torch.nn.functional.softmax(torch.tensor(val_logits), dim=-1)[:, 1].numpy()

    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.01):
        preds = (val_probs >= thresh).astype(int)
        f1 = f1_score(val_labels, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    logger.info(f"Optimal threshold found: {best_thresh:.2f}")
    logger.info(f"Validation Macro F1 with 0.50 threshold : {f1_score(val_labels, np.argmax(val_logits, axis=-1), average='macro'):.4f}")
    logger.info(f"Validation Macro F1 with {best_thresh:.2f} threshold: {best_f1:.4f}")

    # ------------------------------------------------------------------
    # 8. Evaluate on Test Set using Optimized Threshold
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Evaluating on TEST set with optimized threshold")
    logger.info("=" * 60)

    test_output = trainer.predict(tokenized_datasets["test"])
    test_logits = test_output.predictions
    test_labels = test_output.label_ids
    test_probs = torch.nn.functional.softmax(torch.tensor(test_logits), dim=-1)[:, 1].numpy()

    test_results = compute_metrics_with_threshold(test_probs, test_labels, best_thresh)

    logger.info("Test set results (Optimized):")
    for key, value in test_results.items():
        if isinstance(value, float):
            logger.info(f"  {key:25s} : {value:.4f}")
        else:
            logger.info(f"  {key:25s} : {value}")

    trainer.save_metrics("test", test_results)

    # ------------------------------------------------------------------
    # 9. Save model and tokenizer
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Saving final model")
    logger.info("=" * 60)

    trainer.save_model(MODEL_DIR / "final")
    tokenizer.save_pretrained(MODEL_DIR / "final")

    summary = {
        "model": MODEL_NAME,
        "task": "Task A - Binary Sexism Detection",
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "train_samples": len(dataset["train"]),
        "dev_samples": len(dataset["validation"]),
        "test_samples": len(dataset["test"]),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_f1_0.5_threshold": float(trainer.state.best_metric) if trainer.state.best_metric else None,
        "best_dev_f1_optimized_threshold": float(best_f1),
        "optimal_threshold": float(best_thresh),
        "test_f1_macro": float(test_results.get("f1_macro", 0)),
        "test_accuracy": float(test_results.get("accuracy", 0)),
        "test_precision_macro": float(test_results.get("precision_macro", 0)),
        "test_recall_macro": float(test_results.get("recall_macro", 0)),
        "timestamp": timestamp,
    }

    with open(MODEL_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {MODEL_DIR / 'training_summary.json'}")
    logger.info("=" * 60)
    logger.info("Baseline training complete!")
    logger.info("=" * 60)