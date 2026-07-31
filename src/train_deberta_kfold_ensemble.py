#!/usr/bin/env python3
"""
train_deberta_kfold_ensemble.py
K-fold probability ensemble of DAPT -> weighted-finetune DeBERTa-v3-large (EDOS Task A).

Why: single-model recipe tweaks are now inside noise (~0.009 spread). Averaging K
decorrelated fold models reduces variance and combines complementary errors -- the
proven path to +1-2 macro-F1 on this benchmark (it is what the EDOS winner did).

Cleanliness:
  * K stratified folds over the 14k TRAIN. Each fold's 2.8k held-out is used ONLY for
    early-stopping / best-checkpoint selection (logged as a stability signal, mean +/- std).
  * The official 2k DEV is NEVER trained/selected on -> clean hold-out for threshold tuning
    and for the unbiased ensemble score.
  * The official 4k TEST is the single final look.
  * Final prediction = mean of the K folds' softmax probs; threshold tuned on DEV.

Reuses the exact recipe that gave test f1_macro = 0.8593 (fp16, lr 2e-5, weighted CE,
early-stop patience 3). Saves per-fold models + averaged probability vectors so a later
heterogeneous stack (SVM / RF-LR-embed / Qwen) can plug straight in.

Usage (run on the server where the DAPT checkpoint lives):
  CUDA_VISIBLE_DEVICES=0 python src/train_deberta_kfold_ensemble.py \
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
MODEL_DIR = PROJECT_ROOT / "models" / "deberta" / "task_a_kfold_ensemble"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_deberta_kfold_ensemble_{timestamp}.log"
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


def metrics_from_probs(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    pc = f1_score(labels, preds, average=None, zero_division=0)
    return {"accuracy": float((preds == labels).mean()), "f1_macro": float(f1),
            "f1_not_sexist": float(pc[0]) if len(pc) > 0 else 0.0,
            "f1_sexist": float(pc[1]) if len(pc) > 1 else 0.0,
            "n_pred_sexist": int(preds.sum()), "n_pred_not": int((1 - preds).sum())}


def softmax_pos(logits):
    return torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()


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
                   help="Use bf16 instead of the default fp16 (fp16 matches the 0.8593 server run)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MAX_LENGTH, PATIENCE = 256, 3

    # ---- data (tokenize ONCE, then select indices per fold) ----
    logger.info("Loading + tokenizing datasets (once)...")
    ds = load_dataset("csv", data_files={
        "train": str(PROC_DIR / "task_a_train.csv"),
        "validation": str(PROC_DIR / "task_a_dev.csv"),
        "test": str(PROC_DIR / "task_a_test.csv")})
    tok = AutoTokenizer.from_pretrained(args.model_path)
    def tok_fn(ex): return tok(ex["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    ds = ds.map(tok_fn, batched=True)
    train_labels = np.array(ds["train"]["label"])
    logger.info(f"Train={len(ds['train'])}  Dev={len(ds['validation'])}  Test={len(ds['test'])}  "
                f"sexist={int(train_labels.sum())}")

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed_base)
    fold_dev_f1, fold_test_probs, fold_dev_probs = [], [], []
    t_all = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_labels, train_labels)):
        logger.info("=" * 60)
        logger.info(f"FOLD {fold + 1}/{args.k}  train={len(tr_idx)}  val={len(va_idx)}")
        logger.info("=" * 60)
        fold_dir = MODEL_DIR / f"fold_{fold}"
        set_seed(args.seed_base + fold)

        # fresh model every fold (do NOT carry weights across folds)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=2, ignore_mismatched_sizes=False, torch_dtype=torch.float32)

        # class weights from THIS fold's train split
        fl = train_labels[tr_idx]
        n1, n0 = int(fl.sum()), int((fl == 0).sum()); n = len(fl)
        cw = torch.tensor([n / (2 * n0), n / (2 * n1)], dtype=torch.float)
        logger.info(f"Fold class weights: [not={cw[0]:.3f}, sexist={cw[1]:.3f}]")

        ta = TrainingArguments(
            output_dir=str(fold_dir), eval_strategy="epoch", save_strategy="epoch",
            learning_rate=args.lr, per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=32, gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs, weight_decay=0.01, warmup_ratio=0.1,
            load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
            logging_steps=100, seed=args.seed_base + fold,
            fp16=not args.bf16, bf16=args.bf16,
            dataloader_num_workers=4, remove_unused_columns=True, report_to="none")

        def cm(ep):  # per-fold val metric (used for selection -> logged as stability only)
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

        # probabilities from this fold's best model, on clean DEV and on TEST
        fold_dev_probs.append(softmax_pos(trainer.predict(ds["validation"]).predictions))
        fold_test_probs.append(softmax_pos(trainer.predict(ds["test"]).predictions))

        # free VRAM before next fold
        del model, trainer; gc.collect(); torch.cuda.empty_cache()

    # ---- ensemble = mean of fold probabilities ----
    dev_probs = np.mean(fold_dev_probs, axis=0)     # (2000,)  clean hold-out
    test_probs = np.mean(fold_test_probs, axis=0)   # (4000,)  final look
    dev_labels = np.array(ds["validation"]["label"])
    test_labels = np.array(ds["test"]["label"])

    # threshold tuned on the CLEAN dev (never trained/selected on)
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(dev_labels, (dev_probs >= t).astype(int), average="macro")
        if f > best_f: best_f, best_t = f, t
    logger.info("=" * 60); logger.info("ENSEMBLE RESULTS"); logger.info("=" * 60)
    logger.info(f"Per-fold val f1_macro (stability): {np.mean(fold_dev_f1):.4f} +/- {np.std(fold_dev_f1):.4f}")
    logger.info(f"Ensemble threshold (tuned on clean dev): {best_t:.2f}")
    dev_m = metrics_from_probs(dev_probs, dev_labels, best_t)
    logger.info("Ensemble on CLEAN DEV:")
    for k, v in dev_m.items(): logger.info(f"  {k:16s}: {v:.4f}" if isinstance(v, float) else f"  {k:16s}: {v}")
    test_m = metrics_from_probs(test_probs, test_labels, best_t)
    logger.info("Ensemble on TEST (final look):")
    for k, v in test_m.items(): logger.info(f"  {k:16s}: {v:.4f}" if isinstance(v, float) else f"  {k:16s}: {v}")
    logger.info("\n" + classification_report(
        test_labels, (test_probs >= best_t).astype(int),
        target_names=["not_sexist", "sexist"], digits=4))

    # ---- save probs (for a later heterogeneous stack) + summary ----
    np.save(MODEL_DIR / "ensemble_dev_probs.npy", dev_probs)
    np.save(MODEL_DIR / "ensemble_test_probs.npy", test_probs)
    np.save(MODEL_DIR / "ensemble_dev_labels.npy", dev_labels)
    np.save(MODEL_DIR / "ensemble_test_labels.npy", test_labels)
    summary = {"model": args.model_path, "k": args.k, "seed_base": args.seed_base,
               "precision": "bf16" if args.bf16 else "fp16", "lr": args.lr,
               "fold_val_f1_mean": float(np.mean(fold_dev_f1)), "fold_val_f1_std": float(np.std(fold_dev_f1)),
               "threshold": float(best_t),
               "dev_f1_macro": dev_m["f1_macro"], "dev_f1_sexist": dev_m["f1_sexist"],
               "test_f1_macro": test_m["f1_macro"], "test_f1_sexist": test_m["f1_sexist"],
               "test_f1_not_sexist": test_m["f1_not_sexist"], "test_accuracy": test_m["accuracy"],
               "total_seconds": round(time.time() - t_all, 1), "timestamp": timestamp}
    json.dump(summary, open(MODEL_DIR / "ensemble_summary.json", "w"), indent=2)
    logger.info(f"Saved probs + summary to {MODEL_DIR}; total {summary['total_seconds']:.0f}s")
    logger.info("K-fold ensemble complete!")
