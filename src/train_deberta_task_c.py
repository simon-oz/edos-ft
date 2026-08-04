#!/usr/bin/env python3
"""train_deberta_task_c.py
Fine-tune DeBERTa-v3-large (or DAPT-adapted DeBERTa) on EDOS Task C
(fine-grained vector classification). Task-C generalization of train_deberta_task_b.py:
* paths -> task_c_{train,dev,test}.csv
* target names read from the 'label_vector' column (generic class_<i> fallback)
* num_classes detected dynamically (EDOS Task C has 10-11 vectors)

SURGICAL CHANGES (vs previous build):
* --sqrt_weights : use sqrt-damped inverse-frequency class weights
  (w = (n/(C*c))**0.5) instead of full inverse frequency — stabilizes the huge
  weights on 14-21-sample vectors.
* --augment / --augment_below / --augment_target : EDA-style augmentation
  (swap/delete/insert) for classes with fewer than --augment_below train samples,
  upsampled to --augment_target — gives the tiny vectors real signal.

Input: data/processed/task_c_{train,dev,test}.csv
Output: models/deberta/task_c_baseline/ — checkpoints and final model
        logs/train_deberta_task_c_*.log
Usage:
CUDA_VISIBLE_DEVICES=0 python src/train_deberta_task_c.py \
    --model_path /data/pyworkspace/edos-ft/models/deberta/dapt_mlm/final \
    --batch_size 16 --grad_accum 2 --epochs 10 --lr 2e-5 \
    --sqrt_weights --augment

CUDA_VISIBLE_DEVICES=0 python src/train_deberta_task_c.py \
    --model_path models/deberta/dapt_mlm/final \
    --sqrt_weights --augment --lr 2e-5 --epochs 10    
"""
import sys
import json
import logging
import argparse
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from sklearn.metrics import (f1_score, precision_recall_fscore_support, classification_report)
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback, set_seed)

# ------------------------------------------------------------------
# Auto-detect project root
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "deberta" / "task_c_baseline"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_deberta_task_c_{timestamp}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")


# ------------------------------------------------------------------
# Class-weighted trainer
# ------------------------------------------------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *a, **kw):
        super().__init__(*a, **kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.model.config.num_labels), labels.view(-1), weight=w)
        return (loss, outputs) if return_outputs else loss


def compute_metrics_standard(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = float((preds == labels).mean())
    return {"accuracy": acc, "f1_macro": float(f1),
            "precision_macro": float(p), "recall_macro": float(r)}


# ------------------------------------------------------------------
# SURGICAL ADD: light EDA augmentation (no external deps, seedable)
# ------------------------------------------------------------------
def _eda_augment(text, rng, p=0.1):
    words = text.split()
    if len(words) < 4:
        return text
    op = rng.choice(["swap", "delete", "insert"])
    if op == "swap":
        i, j = rng.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    elif op == "delete":
        words = [w for w in words if rng.random() > p] or words
    else:
        for _ in range(max(1, int(p * len(words)))):
            words.insert(rng.randint(0, len(words) - 1), rng.choice(words))
    return " ".join(words)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="microsoft/deberta-v3-large")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    # SURGICAL ADD
    p.add_argument("--sqrt_weights", action="store_true",
                   help="Use sqrt-damped inverse-frequency class weights")
    p.add_argument("--augment", action="store_true",
                   help="EDA-augment classes below --augment_below up to --augment_target")
    p.add_argument("--augment_below", type=int, default=100)
    p.add_argument("--augment_target", type=int, default=150)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MODEL_NAME = args.model_path
    MAX_LENGTH = 256
    SEED = 42
    set_seed(SEED)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("=" * 60); logger.info("Loading Task C datasets"); logger.info("=" * 60)
    train_path, dev_path, test_path = (PROC_DIR / "task_c_train.csv",
                                       PROC_DIR / "task_c_dev.csv",
                                       PROC_DIR / "task_c_test.csv")
    for pth in (train_path, dev_path, test_path):
        if not pth.exists():
            logger.error(f"Missing: {pth}. Run parse_edos_data.py first."); sys.exit(1)
    dataset = load_dataset("csv", data_files={"train": str(train_path),
                                              "validation": str(dev_path),
                                              "test": str(test_path)})
    logger.info(f"Train : {len(dataset['train'])} samples")
    logger.info(f"Validation : {len(dataset['validation'])} samples")
    logger.info(f"Test : {len(dataset['test'])} samples")

    # ------------------------------------------------------------------
    # SURGICAL ADD: EDA augmentation for tiny classes (before labels/weights)
    # ------------------------------------------------------------------
    if args.augment:
        rng = random.Random(SEED)
        cols = dataset["train"].column_names
        train_pdf = pd.DataFrame(dataset["train"])
        counts = train_pdf["label"].value_counts()
        aug_rows = []
        for lab in sorted(train_pdf["label"].unique()):
            n = int(counts[lab])
            if n < args.augment_below:
                sub = train_pdf[train_pdf["label"] == lab]
                for _ in range(max(0, args.augment_target - n)):
                    src = sub.iloc[rng.randrange(len(sub))]
                    row = {c: src[c] for c in cols}
                    row["text"] = _eda_augment(str(src["text"]), rng)
                    row["label"] = int(row["label"])
                    aug_rows.append(row)
        if aug_rows:
            aug_ds = Dataset.from_list(aug_rows, features=dataset["train"].features)
            dataset["train"] = concatenate_datasets([dataset["train"], aug_ds])
            logger.info(f"[augment] added {len(aug_rows)} rows -> train size {len(dataset['train'])}")
    # ------------------------------------------------------------------

    labels_train = np.array(dataset["train"]["label"])
    num_classes = int(labels_train.max()) + 1
    logger.info(f"Detected {num_classes} classes for Task C.")

    # target names from label_vector (generic fallback)
    if "label_vector" in dataset["train"].column_names:
        pairs = list(zip(dataset["train"]["label"], dataset["train"]["label_vector"]))
        label_map = {int(l): v for l, v in pairs}
        target_names = [str(label_map.get(i, f"class_{i}")) for i in range(num_classes)]
    else:
        target_names = [f"class_{i}" for i in range(num_classes)]

    # ------------------------------------------------------------------
    # Class weights (SURGICAL: optional sqrt damping)
    # ------------------------------------------------------------------
    class_counts = np.bincount(labels_train, minlength=num_classes)
    if args.sqrt_weights:
        _w = [(len(labels_train) / (num_classes * c)) ** 0.5 if c > 0 else 0.0 for c in class_counts]
        logger.info("Using sqrt-damped class weights.")
    else:
        _w = [len(labels_train) / (num_classes * c) if c > 0 else 0.0 for c in class_counts]
    class_weights = torch.tensor(_w, dtype=torch.float)
    logger.info(f"Class weights: {class_weights.tolist()}")

    # ------------------------------------------------------------------
    # 2. Tokenize
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length",
                         max_length=MAX_LENGTH)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=False,
        torch_dtype=torch.float32)

    # ------------------------------------------------------------------
    # 4. Training arguments
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        eval_strategy="epoch", save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro", greater_is_better=True,
        warmup_ratio=0.1, logging_steps=50,
        logging_dir=str(PROJECT_ROOT / "logs" / "tensorboard" / "deberta_task_c"),
        report_to=["tensorboard"], seed=SEED, fp16=True,
        dataloader_num_workers=4, remove_unused_columns=True)
    logger.info(f"Model path : {MODEL_NAME}")
    logger.info(f"Learning rate : {args.lr}")
    logger.info(f"Early stopping : patience=3, metric=f1_macro")

    # ------------------------------------------------------------------
    # 5. Trainer
    # ------------------------------------------------------------------
    trainer = WeightedTrainer(class_weights=class_weights, model=model, args=training_args,
                              train_dataset=tokenized_datasets["train"],
                              eval_dataset=tokenized_datasets["validation"],
                              processing_class=tokenizer,
                              compute_metrics=compute_metrics_standard,
                              callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    logger.info("=" * 60); logger.info("Starting training"); logger.info("=" * 60)
    train_result = trainer.train()
    logger.info("Training complete.")
    logger.info(f" Final train loss : {train_result.training_loss:.4f}")
    logger.info(f" Best checkpoint : {trainer.state.best_model_checkpoint}")
    logger.info(f" Best F1 (dev) : {trainer.state.best_metric:.4f}")

    # ------------------------------------------------------------------
    # 7. Test evaluation
    # ------------------------------------------------------------------
    logger.info("=" * 60); logger.info("Evaluating on TEST"); logger.info("=" * 60)
    test_out = trainer.predict(tokenized_datasets["test"])
    test_preds = np.argmax(test_out.predictions, axis=-1)
    test_labels = test_out.label_ids
    test_f1 = float(f1_score(test_labels, test_preds, average="macro", zero_division=0))
    test_acc = float((test_preds == test_labels).mean())
    logger.info("TEST classification report:")
    logger.info("\n" + classification_report(test_labels, test_preds,
                                             target_names=target_names, digits=4, zero_division=0))

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    logger.info("=" * 60); logger.info("Saving model"); logger.info("=" * 60)
    trainer.save_model(MODEL_DIR / "final")
    tokenizer.save_pretrained(MODEL_DIR / "final")
    summary = {"model": MODEL_NAME, "task": "Task C - Fine-Grained Sexism Vector Detection",
               "seed": SEED, "max_length": MAX_LENGTH, "num_classes": num_classes,
               "target_names": target_names,
               "sqrt_weights": args.sqrt_weights, "augment": args.augment,
               "train_samples": len(dataset["train"]),
               "dev_samples": len(dataset["validation"]),
               "test_samples": len(dataset["test"]),
               "best_checkpoint": trainer.state.best_model_checkpoint,
               "best_dev_f1": float(trainer.state.best_metric) if trainer.state.best_metric else None,
               "test_f1_macro": test_f1, "test_accuracy": test_acc,
               "timestamp": timestamp}
    with open(MODEL_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {MODEL_DIR / 'training_summary.json'}")
    logger.info("=" * 60); logger.info("Task C training complete!"); logger.info("=" * 60)