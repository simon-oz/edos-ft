#!/usr/bin/env python3
"""
train_deberta_baseline_rtx3060.py  (v4 - CLEAN)
Train DeBERTa-v3-large on EDOS Task A.
Uses STANDARD HuggingFace Trainer — no custom loss.
Matches EDOS paper hyperparameters (PingAnLifeInsurance, F1=0.8746).

Usage:
  CUDA_VISIBLE_DEVICES=0 python src/train_deberta_baseline_rtx3060.py \
      --model_path /media/simon/node70-data11/Models/microsoft/deberta-v3-large
"""
import sys, json, logging, os, time, argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
    classification_report, accuracy_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)


# ======================================================================
# Sanity checks (Phase 0)
# ======================================================================
def run_sanity_checks(dataset, model, tokenizer, logger):
    """Run BEFORE training. Catches data/model issues in < 30 seconds."""
    logger.info("=" * 60)
    logger.info("PRE-TRAINING SANITY CHECKS")
    logger.info("=" * 60)

    # ── Check 1: Data format ──
    train = dataset["train"]
    logger.info(f"[Check 1] Column names: {train.column_names}")
    assert "text" in train.column_names, "Missing 'text' column!"
    assert "label" in train.column_names, "Missing 'label' column!"

    # ── Check 2: Label types and values ──
    labels = train["label"]
    unique_labels = set(labels)
    logger.info(f"[Check 2] Unique labels: {unique_labels}")
    logger.info(f"[Check 2] Label dtype: {type(labels[0])}")
    assert unique_labels == {0, 1}, f"Expected {{0, 1}}, got {unique_labels}"

    # ── Check 3: First 5 samples ──
    for i in range(5):
        text_preview = train[i]["text"][:80]
        label = train[i]["label"]
        logger.info(f"[Check 3] Sample {i}: label={label}, text='{text_preview}...'")

    # ── Check 4: Class distribution ──
    n_sexist = sum(labels)
    n_total = len(labels)
    logger.info(f"[Check 4] Distribution: sexist={n_sexist} ({n_sexist/n_total:.1%}), "
                f"not_sexist={n_total - n_sexist} ({(n_total-n_sexist)/n_total:.1%})")

    # ── Check 5: Initial model logits ──
    logger.info("[Check 5] Testing initial model output on 5 samples...")
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(5):
            inputs = tokenizer(
                train[i]["text"],
                truncation=True, padding="max_length", max_length=256,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits[0]
            pred = logits.argmax().item()
            probs = torch.softmax(logits, dim=-1)
            logger.info(
                f"  Sample {i} (true={train[i]['label']}): "
                f"logits=[{logits[0]:.3f}, {logits[1]:.3f}], "
                f"probs=[{probs[0]:.4f}, {probs[1]:.4f}], "
                f"pred={'sexist' if pred == 1 else 'not_sexist'}"
            )
    model.train()

    # ── Check 6: Tokenization ──
    sample_tokens = tokenizer(train[0]["text"], truncation=True, max_length=256)
    logger.info(f"[Check 6] Token count for sample 0: {len(sample_tokens['input_ids'])}")
    logger.info(f"[Check 6] First 10 token IDs: {sample_tokens['input_ids'][:10]}")

    logger.info("=" * 60)
    logger.info("All sanity checks PASSED")
    logger.info("=" * 60)


# ======================================================================
# Metrics with collapse detection
# ======================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    n_pred_sexist = int(np.sum(preds))
    n_pred_not = len(preds) - n_pred_sexist

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_not_sexist": float(per_class_f1[0]) if len(per_class_f1) > 0 else 0.0,
        "f1_sexist": float(per_class_f1[1]) if len(per_class_f1) > 1 else 0.0,
        "n_pred_sexist": n_pred_sexist,
        "n_pred_not_sexist": n_pred_not,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="microsoft/deberta-v3-large")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.2e-5,
                        help="EDOS paper winner used 1.2e-5")
    parser.add_argument("--warmup_ratio", type=float, default=0.10,
                        help="EDOS paper used 10%% warmup")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--early_stopping", action="store_true", default=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--skip_sanity", action="store_true", default=False)
    args = parser.parse_args()

    # ── Paths ──
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
    PROC_DIR = PROJECT_ROOT / "data" / "processed"
    MODEL_DIR = PROJECT_ROOT / "models" / "deberta" / "task_a_v4"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Logging ──
    LOG_DIR = PROJECT_ROOT / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"train_deberta_v4_{timestamp}.log"

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

    # ── GPU ──
    if not torch.cuda.is_available():
        logger.error("CUDA not available.")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info(f"GPU detected: {gpu_name}")
    logger.info(f"GPU memory  : {gpu_mem:.1f} GB")

    # ── Config ──
    MODEL_NAME = args.model_path
    MAX_LENGTH = 256
    SEED = 42
    set_seed(SEED)

    logger.info("=" * 60)
    logger.info("Configuration (v4 - CLEAN, EDOS paper settings)")
    logger.info("=" * 60)
    logger.info(f"Model path         : {MODEL_NAME}")
    logger.info(f"Max sequence length: {MAX_LENGTH}")
    logger.info(f"Per-device batch   : {args.batch_size}")
    logger.info(f"Gradient accum     : {args.grad_accum}")
    logger.info(f"Effective batch    : {args.batch_size * args.grad_accum}")
    logger.info(f"Precision          : bf16")
    logger.info(f"Max epochs         : {args.epochs}")
    logger.info(f"Learning rate      : {args.lr}")
    logger.info(f"Warmup ratio       : {args.warmup_ratio}")
    logger.info(f"Max grad norm      : {args.max_grad_norm}")
    logger.info(f"Loss function      : Standard CrossEntropy (model internal)")
    logger.info(f"Early stopping     : "
                f"{'patience=3' if args.early_stopping else 'DISABLED'}")

    # ── Load data ──
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

    dataset = load_dataset("csv", data_files={
        "train": str(train_path),
        "validation": str(dev_path),
        "test": str(test_path),
    })

    logger.info(f"Train      : {len(dataset['train'])} samples")
    logger.info(f"Validation : {len(dataset['validation'])} samples")
    logger.info(f"Test       : {len(dataset['test'])} samples")

    labels_train = dataset["train"]["label"]
    n_sexist = sum(labels_train)
    n_not_sexist = len(labels_train) - n_sexist
    logger.info(f"Train class distribution: sexist={n_sexist}, not_sexist={n_not_sexist}")

    # ── Tokenize ──
    logger.info("=" * 60)
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # ── Load model ──
    logger.info("=" * 60)
    logger.info(f"Loading model: {MODEL_NAME}")
    logger.info("=" * 60)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        local_files_only=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters     : {total_params:,}")
    logger.info(f"Trainable parameters : {trainable_params:,}")

    # ── Sanity checks ──
    if not args.skip_sanity:
        run_sanity_checks(dataset, model, tokenizer, logger)

    # ── Training arguments (EDOS paper config) ──
    logger.info("=" * 60)
    logger.info("Configuring training")
    logger.info("=" * 60)

    steps_per_epoch = len(dataset["train"]) // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,                    # 1.2e-5 (EDOS paper)
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        max_grad_norm=args.max_grad_norm,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        warmup_steps=warmup_steps,                # 10% (EDOS paper)
        logging_steps=50,
        report_to=["tensorboard"],
        seed=SEED,
        bf16=True,
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True if args.workers > 0 else False,
        remove_unused_columns=True,
        # NO label_smoothing_factor — use model's internal CE loss
        # NO custom loss — use standard Trainer
    )

    logger.info(f"Effective batch size : {args.batch_size * args.grad_accum}")
    logger.info(f"Warmup steps         : {warmup_steps}")
    logger.info(f"Total steps          : {total_steps}")

    # ── Trainer (STANDARD — no custom loss) ──
    callbacks = []
    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ── Train ──
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)

    train_start = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - train_start

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info(f"  Final train loss : {train_result.training_loss:.4f}")
    logger.info(f"  Best checkpoint  : {trainer.state.best_model_checkpoint}")
    logger.info(f"  Best F1 (dev)    : {trainer.state.best_metric:.4f}")
    logger.info(f"  Total train time : {train_seconds:.1f}s ({train_seconds/60:.1f} min)")
    logger.info("=" * 60)

    train_metrics = train_result.metrics
    train_metrics["total_train_time_seconds"] = round(train_seconds, 1)
    trainer.save_metrics("train", train_metrics)

    # ── Evaluate on test set ──
    logger.info("=" * 60)
    logger.info("Evaluating on TEST set")
    logger.info("=" * 60)

    test_results = trainer.evaluate(tokenized_datasets["test"])
    for key, value in test_results.items():
        if isinstance(value, float):
            logger.info(f"  {key:25s} : {value:.4f}")

    test_preds = trainer.predict(tokenized_datasets["test"])
    test_labels = test_preds.label_ids
    test_predictions = np.argmax(test_preds.predictions, axis=-1)

    n_pred_sexist = int(np.sum(test_predictions))
    n_pred_not = len(test_predictions) - n_pred_sexist
    logger.info(f"  Prediction distribution: not_sexist={n_pred_not}, sexist={n_pred_sexist}")

    logger.info("\nDetailed classification report (TEST):")
    report = classification_report(
        test_labels, test_predictions,
        target_names=["not_sexist", "sexist"],
        digits=4,
    )
    for line in report.split("\n"):
        logger.info(line)

    # ── Save ──
    logger.info("=" * 60)
    logger.info("Saving final model")
    logger.info("=" * 60)

    trainer.save_model(MODEL_DIR / "final")
    tokenizer.save_pretrained(MODEL_DIR / "final")

    summary = {
        "model": MODEL_NAME,
        "task": "Task A - Binary Sexism Detection",
        "script_version": "v4_clean",
        "gpu": gpu_name,
        "gpu_memory_gb": round(gpu_mem, 1),
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "precision": "bf16",
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "loss_function": "standard_cross_entropy",
        "train_samples": len(dataset["train"]),
        "dev_samples": len(dataset["validation"]),
        "test_samples": len(dataset["test"]),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_f1_macro": float(trainer.state.best_metric) if trainer.state.best_metric else None,
        "test_f1_macro": float(test_results.get("eval_f1_macro", 0)),
        "test_accuracy": float(test_results.get("eval_accuracy", 0)),
        "test_f1_sexist": float(test_results.get("eval_f1_sexist", 0)),
        "test_f1_not_sexist": float(test_results.get("eval_f1_not_sexist", 0)),
        "total_train_time_seconds": round(train_seconds, 1),
        "timestamp": timestamp,
    }

    with open(MODEL_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {MODEL_DIR / 'training_summary.json'}")
    logger.info("=" * 60)
    logger.info("v4 training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()