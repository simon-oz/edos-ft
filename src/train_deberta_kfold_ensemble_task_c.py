#!/usr/bin/env python3
"""train_deberta_kfold_ensemble_task_c.py
K-fold probability ensemble of (DAPT-adapted) DeBERTa-v3-large on EDOS Task C
(fine-grained vector classification). Task-C generalization of
train_deberta_kfold_ensemble_task_b.py:
  * softmax returns the FULL (N, C) probability matrix
  * ensemble prediction = ARGMAX over the mean (N, C) probs
  * per-fold class weights are C-way: n / (num_classes * count_c)
  * target names read from 'label_vector' (generic fallback); num_classes dynamic

Cleanliness (same philosophy as Task A/B):
  * K stratified folds over TRAIN; each fold's held-out used ONLY for selection.
  * Official DEV never trained/selected on -> clean hold-out for the ensemble score.
  * Official TEST is the single final look.

IMPORTANT: probability vectors are saved into THIS script's own MODEL_DIR, NOT into
models/ensemble_probs/ (that folder is globbed by the BINARY ensemble_vote.py, which
would crash on a multi-column array). A Task C heterogeneous voter needs its own
multi-class voter (argmax over (N,C)).

Usage:
  CUDA_VISIBLE_DEVICES=0 python src/train_deberta_kfold_ensemble_task_c.py \
      --model_path models/deberta/dapt_mlm/final --k 5
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
MODEL_DIR = PROJECT_ROOT / "models" / "deberta" / "task_c_kfold_ensemble"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_deberta_kfold_ensemble_task_c_{timestamp}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")


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


def softmax_all(logits):
    return torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()


def metrics_from_probs(probs, labels, num_classes):
    preds = probs.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    pc = f1_score(labels, preds, average=None, zero_division=0)
    return {"accuracy": float((preds == labels).mean()), "f1_macro": float(f1),
            "per_class_f1": [float(x) for x in pc],
            "pred_dist": np.bincount(preds, minlength=num_classes).tolist()}


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
    p.add_argument("--bf16", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MAX_LENGTH, PATIENCE = 256, 3

    logger.info("Loading + tokenizing Task C datasets (once)...")
    ds = load_dataset("csv", data_files={"train": str(PROC_DIR / "task_c_train.csv"),
                                         "validation": str(PROC_DIR / "task_c_dev.csv"),
                                         "test": str(PROC_DIR / "task_c_test.csv")})
    tok = AutoTokenizer.from_pretrained(args.model_path)
    def tok_fn(ex): return tok(ex["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    ds = ds.map(tok_fn, batched=True)
    train_labels = np.array(ds["train"]["label"], dtype=np.int64)

    num_classes = int(train_labels.max()) + 1
    uniq = np.unique(train_labels)
    if not (uniq.min() >= 0 and uniq.max() == num_classes - 1 and len(uniq) == num_classes):
        logger.warning(f"Labels may not be contiguous 0..{num_classes-1}; got unique={uniq.tolist()}.")

    if "label_vector" in ds["train"].column_names:
        pairs = list(zip(ds["train"]["label"], ds["train"]["label_vector"]))
        label_map = {int(l): v for l, v in pairs}
        target_names = [label_map[i] for i in sorted(label_map)]
    else:
        target_names = [f"class_{i}" for i in range(num_classes)]
    logger.info(f"Train={len(ds['train'])} Dev={len(ds['validation'])} Test={len(ds['test'])} "
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

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=num_classes, ignore_mismatched_sizes=False,
            torch_dtype=torch.float32)

        fl = train_labels[tr_idx]; n = len(fl)
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

        def cm(ep):
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

        fold_dev_probs.append(softmax_all(trainer.predict(ds["validation"]).predictions))
        fold_test_probs.append(softmax_all(trainer.predict(ds["test"]).predictions))

        del model, trainer; gc.collect(); torch.cuda.empty_cache()

    dev_probs = np.mean(fold_dev_probs, axis=0)
    test_probs = np.mean(fold_test_probs, axis=0)
    dev_labels = np.array(ds["validation"]["label"], dtype=np.int64)
    test_labels = np.array(ds["test"]["label"], dtype=np.int64)

    logger.info("=" * 60); logger.info("ENSEMBLE RESULTS (argmax over mean probs)"); logger.info("=" * 60)
    logger.info(f"Per-fold val f1_macro (stability): {np.mean(fold_dev_f1):.4f} +/- {np.std(fold_dev_f1):.4f}")
    dev_m = metrics_from_probs(dev_probs, dev_labels, num_classes)
    log_metrics("Ensemble on CLEAN DEV", dev_m, target_names)
    test_m = metrics_from_probs(test_probs, test_labels, num_classes)
    log_metrics("Ensemble on TEST (final look)", test_m, target_names)
    logger.info("\n" + classification_report(test_labels, test_probs.argmax(axis=1),
                                             target_names=target_names, digits=4, zero_division=0))

    np.save(MODEL_DIR / "ensemble_dev_probs.npy", dev_probs)
    np.save(MODEL_DIR / "ensemble_test_probs.npy", test_probs)
    np.save(MODEL_DIR / "ensemble_dev_labels.npy", dev_labels)
    np.save(MODEL_DIR / "ensemble_test_labels.npy", test_labels)
    summary = {"model": args.model_path, "task": "Task C", "k": args.k,
               "num_classes": num_classes, "target_names": target_names,
               "seed_base": args.seed_base, "precision": "bf16" if args.bf16 else "fp16", "lr": args.lr,
               "fold_val_f1_mean": float(np.mean(fold_dev_f1)), "fold_val_f1_std": float(np.std(fold_dev_f1)),
               "dev_f1_macro": dev_m["f1_macro"], "test_f1_macro": test_m["f1_macro"],
               "test_accuracy": test_m["accuracy"],
               "total_seconds": round(time.time() - t_all, 1), "timestamp": timestamp}
    json.dump(summary, open(MODEL_DIR / "ensemble_summary.json", "w"), indent=2)
    logger.info(f"Saved probs + summary to {MODEL_DIR}; total {summary['total_seconds']:.0f}s")
    logger.info("Task C K-fold ensemble complete!")