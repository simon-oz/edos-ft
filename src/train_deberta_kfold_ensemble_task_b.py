#!/usr/bin/env python3
"""
train_deberta_kfold_ensemble_task_b.py
K-fold probability ensemble of (DAPT-adapted) DeBERTa-v3-large on EDOS Task B
(4-category sexism classification). This is the Task-B generalization of
train_deberta_kfold_ensemble.py (which was binary / Task A).

Multi-class generalizations vs the Task A script:
  * softmax returns the FULL (N, 4) probability matrix, not a single P(sexist).
  * Ensemble prediction = ARGMAX over the mean (N,4) probs (a single binary threshold
    does not generalize to 4 classes; argmax is what the Task B baseline uses).
  * Per-fold class weights are 4-way: n / (num_classes * count_c).
  * Metrics / classification report use 4 target names.

Cleanliness (same philosophy as Task A):
  * K stratified folds over the TRAIN split. Each fold's held-out is used ONLY for
    early-stopping / best-checkpoint selection (logged as a stability signal, mean +/- std).
  * The official DEV is NEVER trained/selected on -> clean hold-out for the ensemble score.
  * The official TEST is the single final look.
  * Final prediction = argmax of the mean of the K folds' softmax probs.

IMPORTANT: probability vectors are saved into THIS script's own MODEL_DIR, NOT into
models/ensemble_probs/ (that folder is globbed by the BINARY ensemble_vote.py, which
would crash on a 4-column array). A Task B heterogeneous voter needs its own multi-class
voter (argmax over (N,4)) -- ask and it will be provided.

Usage:
  CUDA_VISIBLE_DEVICES=0 python src/train_deberta_kfold_ensemble_task_b.py \
      --model_path models/deberta/dapt_mlm/final --k 5
  # (or --model_path microsoft/deberta-v3-large to skip DAPT)
"""
import sys, gc, json, logging, argparse, time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, set_seed,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "deberta" / "task_b_kfold_ensemble"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_deberta_kfold_ensemble_task_b_{timestamp}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")

# EDOS Task B category names (label index -> readable name); generic fallback if != 4.
EDOS_B_NAMES = {
    0: "1. threats, plans to harm and incitement",
    1: "2. derogation",
    2: "3. animosity",
    3: "4. prejudiced discussions",
}


# ------------------------------------------------------------------ class-weighted trainer
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *a, **kw):
        super().__init__(*a, **kw); self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs); logits = outputs.logits
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.model.config.num_labels), labels.view(-1), weight=w)
        return (loss, outputs) if return_outputs else loss


# ------------------------------------------------------------------ helpers
def softmax_all(logits):
    """Full softmax over all classes -> shape (N, num_classes)."""
    return torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()


def metrics_from_probs(probs, labels, num_classes):
    """probs: (N, C). Prediction = argmax (no threshold for multi-class)."""
    preds = probs.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    pc = f1_score(labels, preds, average=None, zero_division=0)
    acc = float((preds == labels).mean())
    dist = np.bincount(preds, minlength=num_classes).tolist()
    return {"accuracy": acc, "f1_macro": float(f1),
            "per_class_f1": [float(x) for x in pc], "pred_dist": dist}


def log_metrics(title, m, target_names):
    logger.info(f"{title}:")
    logger.info(f"  accuracy        : {m['accuracy']:.4f}")
    logger.info(f"  f1_macro        : {m['f1_macro']:.4f}")
    for i, fv in enumerate(m["per_class_f1"]):
        name = target_names[i] if i < len(target_names) else f"class_{i}"
        logger.info(f"    f1[{name}] = {fv:.4f}")
    logger.info(f"  pred distribution (per class): {m['pred_dist']}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="models/deberta/dapt_mlm/final")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--bf16", action="store_true", default=False,
                   help="Use bf16 instead of the default fp16")
    return p.parse_args()


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    args = parse_args()
    MAX_LENGTH, PATIENCE = 256, 3

    # ---- data (tokenize ONCE, then select indices per fold) ----
    logger.info("Loading + tokenizing Task B datasets (once)...")
    ds = load_dataset("csv", data_files={
        "train": str(PROC_DIR / "task_b_train.csv"),
        "validation": str(PROC_DIR / "task_b_dev.csv"),
        "test": str(PROC_DIR / "task_b_test.csv")})
    tok = AutoTokenizer.from_pretrained(args.model_path)
    def tok_fn(ex): return tok(ex["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    ds = ds.map(tok_fn, batched=True)
    train_labels = np.array(ds["train"]["label"], dtype=np.int64)

    # ---- number of classes + target names (0-indexed labels assumed, as in train_deberta_task_b.py) ----
    num_classes = int(train_labels.max()) + 1
    uniq = np.unique(train_labels)
    if not (uniq.min() >= 0 and uniq.max() == num_classes - 1 and len(uniq) == num_classes):
        logger.warning(f"Labels may not be contiguous 0..{num_classes-1}; got unique={uniq.tolist()}. "
                       f"CrossEntropyLoss expects 0-indexed targets.")
    target_names = [EDOS_B_NAMES.get(i, f"class_{i}") for i in range(num_classes)]
    logger.info(f"Train={len(ds['train'])}  Dev={len(ds['validation'])}  Test={len(ds['test'])}  "
                f"num_classes={num_classes}")
    logger.info(f"Train class distribution: {np.bincount(train_labels, minlength=num_classes).tolist()}")
    logger.info(f"Target names: {target_names}")

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed_base)
    fold_dev_f1, fold_test_probs, fold_dev_probs = [], [], []
    t_all = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_labels, train_labels)):
        logger.info("=" * 60)
        logger.info(f"FOLD {fold + 1}/{args.k}  train={len(tr_idx)}  val={len(va_idx)}")
        logger.info("=" * 60)
        fold_dir = MODEL_DIR / f"fold_{fold}"
        set_seed(args.seed_base + fold)

        # fresh model every fold (num_labels = num_classes)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=num_classes, ignore_mismatched_sizes=False,
            torch_dtype=torch.float32)

        # 4-way class weights from THIS fold's train split (guard against empty class)
        fl = train_labels[tr_idx]
        n = len(fl)
        counts = np.bincount(fl, minlength=num_classes)
        cw = torch.tensor([n / (num_classes * c) if c > 0 else 0.0 for c in counts], dtype=torch.float)
        logger.info(f"Fold class counts : {counts.tolist()}")
        logger.info(f"Fold class weights: {cw.tolist()}")

        ta = TrainingArguments(
            output_dir=str(fold_dir), eval_strategy="epoch", save_strategy="epoch",
            learning_rate=args.lr, per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=32, gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs, weight_decay=0.01, warmup_ratio=0.1,
            load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
            logging_steps=100, seed=args.seed_base + fold,
            fp16=not args.bf16, bf16=args.bf16,
            dataloader_num_workers=4, remove_unused_columns=True, report_to="none")

        def cm(ep):  # per-fold val metric (selection only -> logged as stability)
            logits, labels = ep; preds = np.argmax(logits, axis=-1)
            return {"f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0))}

        trainer = WeightedTrainer(
            class_weights=cw, model=model, args=ta,
            train_dataset=ds["train"].select(tr_idx.tolist()),
            eval_dataset=ds["train"].select(va_idx.tolist()),
            processing_class=tok, compute_metrics=cm,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)])
        trainer.train()
        fold_dev_f1.append(float(trainer.state.best_metric))
        logger.info(f"Fold {fold + 1} best val f1_macro (selection score) = {fold_dev_f1[-1]:.4f}")

        # FULL (N, C) probabilities from this fold's best model, on clean DEV and TEST
        fold_dev_probs.append(softmax_all(trainer.predict(ds["validation"]).predictions))
        fold_test_probs.append(softmax_all(trainer.predict(ds["test"]).predictions))

        del model, trainer; gc.collect(); torch.cuda.empty_cache()

    # ---- ensemble = mean of fold probability matrices, then argmax ----
    dev_probs = np.mean(fold_dev_probs, axis=0)     # (Ndev, C)
    test_probs = np.mean(fold_test_probs, axis=0)   # (Ntest, C)
    dev_labels = np.array(ds["validation"]["label"], dtype=np.int64)
    test_labels = np.array(ds["test"]["label"], dtype=np.int64)

    logger.info("=" * 60); logger.info("ENSEMBLE RESULTS (argmax over mean probs)"); logger.info("=" * 60)
    logger.info(f"Per-fold val f1_macro (stability): {np.mean(fold_dev_f1):.4f} +/- {np.std(fold_dev_f1):.4f}")
    dev_m = metrics_from_probs(dev_probs, dev_labels, num_classes)
    log_metrics("Ensemble on CLEAN DEV", dev_m, target_names)
    test_m = metrics_from_probs(test_probs, test_labels, num_classes)
    log_metrics("Ensemble on TEST (final look)", test_m, target_names)
    logger.info("\n" + classification_report(
        test_labels, test_probs.argmax(axis=1), target_names=target_names, digits=4, zero_division=0))

    # ---- save probs + labels (into THIS dir only; NOT the binary voter folder) + summary ----
    np.save(MODEL_DIR / "ensemble_dev_probs.npy", dev_probs)     # (Ndev, C)
    np.save(MODEL_DIR / "ensemble_test_probs.npy", test_probs)   # (Ntest, C)
    np.save(MODEL_DIR / "ensemble_dev_labels.npy", dev_labels)
    np.save(MODEL_DIR / "ensemble_test_labels.npy", test_labels)
    summary = {"model": args.model_path, "task": "Task B", "k": args.k,
               "num_classes": num_classes, "target_names": target_names,
               "seed_base": args.seed_base, "precision": "bf16" if args.bf16 else "fp16", "lr": args.lr,
               "fold_val_f1_mean": float(np.mean(fold_dev_f1)), "fold_val_f1_std": float(np.std(fold_dev_f1)),
               "dev_f1_macro": dev_m["f1_macro"], "dev_per_class_f1": dev_m["per_class_f1"],
               "test_f1_macro": test_m["f1_macro"], "test_per_class_f1": test_m["per_class_f1"],
               "test_accuracy": test_m["accuracy"],
               "total_seconds": round(time.time() - t_all, 1), "timestamp": timestamp}
    json.dump(summary, open(MODEL_DIR / "ensemble_summary.json", "w"), indent=2)
    logger.info(f"Saved probs + summary to {MODEL_DIR}; total {summary['total_seconds']:.0f}s")
    logger.info("Task B K-fold ensemble complete!")