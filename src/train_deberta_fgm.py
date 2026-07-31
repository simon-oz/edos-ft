#!/usr/bin/env python3
"""
train_deberta_fgm.py
DeBERTa-v3-large on EDOS Task A with FGM (Fast Gradient Method) adversarial training.

This is the working DAPT->weighted-finetune pipeline (train_deberta_baseline.py)
plus FGM, to attack the overfitting seen at test f1_macro=0.8593.

Deliberate changes vs the 0.8593 baseline (keep ablation clean):
  1. +FGM adversarial training (override training_step: clean + perturbed backward)
  2. fp16 -> bf16 (no GradScaler; fp32 master grads -> stable FGM; native on H100)
Everything else (lr, wd, batch, dropout, class weights, threshold tune) is unchanged.

Typical flow:
  1) DAPT once:        python src/train_deberta_dapt.py --model_path <base>
  2) FGM on DAPT ckpt: python src/train_deberta_fgm.py  --model_path models/deberta/dapt_mlm/final
  3) Ablation (bf16 only, no FGM):
                       python src/train_deberta_fgm.py  --model_path models/deberta/dapt_mlm/final --no_fgm

H100 NVL note: defaults match the baseline (batch 16) for a clean comparison.
FGM doubles forward/backward cost per step, so if you want to use the 94 GB headroom,
bump --batch_size 32 or 64 (and/or --grad_accum) -- gradients stay correctly averaged.
If your box exposes more than one GPU, prefix with CUDA_VISIBLE_DEVICES=0.
"""
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, classification_report,
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
# Project paths / logging
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "deberta" / "task_a_fgm"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_deberta_fgm_{timestamp}.log"
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
# FGM (Miyato et al., 2021): perturb embeddings along grad direction
# ------------------------------------------------------------------
class FGM:
    def __init__(self, model, emb_name="word_embeddings", epsilon=1.0):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm   # unit direction * eps
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ------------------------------------------------------------------
# Class-weighted loss (identical to the working baseline)
# ------------------------------------------------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")                 # mutates the dict -> callers must pass a copy
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ------------------------------------------------------------------
# FGM trainer: clean backward + perturbed backward per micro-batch
# ------------------------------------------------------------------
class FGMTrainer(WeightedTrainer):
    def __init__(self, use_fgm=True, epsilon=1.0, emb_name="word_embeddings", **kwargs):
        super().__init__(**kwargs)
        self.use_fgm = use_fgm
        self.fgm = FGM(self.model, emb_name=emb_name, epsilon=epsilon) if use_fgm else None
        self._last_adv_loss = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not self.use_fgm:
            return super().training_step(model, inputs, num_items_in_batch)

        model.train()
        inputs = self._prepare_inputs(inputs)

        # loss-averaging factor across gradient-accumulation micro-batches.
        # (Custom compute_loss returns a LOCAL mean, so we must divide unless the
        #  model path already normalised by num_items_in_batch.)
        ga = self.args.gradient_accumulation_steps
        scale = 1.0 / ga if not getattr(self, "model_accepts_loss_kwargs", False) else 1.0

        # 1) clean forward + backward
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, dict(inputs), num_items_in_batch=num_items_in_batch)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss * scale)

        # 2) adversarial forward + backward at perturbed embedding, then restore
        self._last_adv_loss = None
        try:
            self.fgm.attack()
            with self.compute_loss_context_manager():
                loss_adv = self.compute_loss(model, dict(inputs), num_items_in_batch=num_items_in_batch)
            if self.args.n_gpu > 1:
                loss_adv = loss_adv.mean()
            self.accelerator.backward(loss_adv * scale)
            self._last_adv_loss = loss_adv.detach()
        finally:
            self.fgm.restore()

        return loss.detach() * scale   # ga=1 -> equals the clean loss (matches baseline logging)

    def log(self, logs, *args, **kwargs):
        # surface the adversarial loss so we can verify FGM is actually doing work
        if self._last_adv_loss is not None:
            logs["adv_loss"] = round(float(self._last_adv_loss), 4)
        super().log(logs, *args, **kwargs)


# ------------------------------------------------------------------
# Metrics / tokenization (identical to baseline)
# ------------------------------------------------------------------
def compute_metrics_standard(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    n_pred_sexist = int(np.sum(preds))
    n_pred_not = len(preds) - n_pred_sexist
    logger.info(f"  [Diag] Predictions: not_sexist={n_pred_not}, sexist={n_pred_sexist}")
    if n_pred_sexist == 0 or n_pred_not == 0:
        logger.warning(f"  !! MODEL COLLAPSED: one class only "
                       f"(not_sexist={n_pred_not}, sexist={n_pred_sexist})")
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    acc = (preds == labels).mean()
    per_class = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_not_sexist": per_class[2][0] if len(per_class[2]) > 0 else 0.0,
        "f1_sexist": per_class[2][1] if len(per_class[2]) > 1 else 0.0,
    }


def compute_metrics_with_threshold(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    acc = (preds == labels).mean()
    per_class = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    return {
        "accuracy": acc, "f1_macro": f1,
        "precision_macro": precision, "recall_macro": recall,
        "f1_not_sexist": per_class[2][0] if len(per_class[2]) > 0 else 0.0,
        "f1_sexist": per_class[2][1] if len(per_class[2]) > 1 else 0.0,
    }


def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(examples["text"], truncation=True,
                     padding="max_length", max_length=max_length)


# ------------------------------------------------------------------
# Main (under guard so dataloader_num_workers>0 is safe)
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="microsoft/deberta-v3-large",
                   help="Use the DAPT checkpoint here, e.g. models/deberta/dapt_mlm/final")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    # FGM-specific
    p.add_argument("--no_fgm", action="store_true",
                   help="Disable FGM -> reproduces the bf16 baseline for ablation")
    p.add_argument("--epsilon", type=float, default=1.0,
                   help="FGM perturbation magnitude (unit-norm direction * eps). Try 0.5-1.0")
    p.add_argument("--emb_name", type=str, default="word_embeddings",
                   help="Substring of the embedding param(s) to perturb")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MODEL_NAME = args.model_path
    MAX_LENGTH = 256
    SEED = 42
    set_seed(SEED)
    USE_FGM = not args.no_fgm

    if not torch.cuda.is_available():
        logger.error("CUDA not available.")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")

    logger.info("=" * 60)
    logger.info("Configuration (FGM build)")
    logger.info("=" * 60)
    logger.info(f"Model path      : {MODEL_NAME}")
    logger.info(f"FGM enabled     : {USE_FGM}")
    logger.info(f"FGM epsilon     : {args.epsilon}")
    logger.info(f"FGM emb_name    : {args.emb_name}")
    logger.info(f"Precision       : bf16")
    logger.info(f"Batch / accum   : {args.batch_size} / {args.grad_accum} "
                f"(effective {args.batch_size * args.grad_accum})")
    logger.info(f"LR / epochs     : {args.lr} / {args.epochs}")

    # 1. data ----------------------------------------------------------------
    logger.info("=" * 60); logger.info("Loading Task A datasets"); logger.info("=" * 60)
    train_path, dev_path, test_path = (PROC_DIR / "task_a_train.csv",
                                       PROC_DIR / "task_a_dev.csv",
                                       PROC_DIR / "task_a_test.csv")
    for pth in (train_path, dev_path, test_path):
        if not pth.exists():
            logger.error(f"Missing: {pth}. Run parse_edos_data.py first."); sys.exit(1)
    dataset = load_dataset("csv", data_files={
        "train": str(train_path), "validation": str(dev_path), "test": str(test_path)})
    logger.info(f"Train={len(dataset['train'])}  Dev={len(dataset['validation'])}  "
                f"Test={len(dataset['test'])}")
    labels_train = np.array(dataset["train"]["label"])
    n_sexist, n_not = int(sum(labels_train)), int(len(labels_train) - sum(labels_train))
    logger.info(f"Train class distribution: sexist={n_sexist}, not_sexist={n_not}")
    w0 = len(labels_train) / (2 * n_not)
    w1 = len(labels_train) / (2 * n_sexist)
    class_weights = torch.tensor([w0, w1], dtype=torch.float)
    logger.info(f"Class weights: [not_sexist={w0:.4f}, sexist={w1:.4f}]")

    # 2. tokenize ------------------------------------------------------------
    logger.info("=" * 60); logger.info(f"Loading tokenizer: {MODEL_NAME}"); logger.info("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH), batched=True)

    # 3. model (fp32 master weights -> fp32 grads -> stable FGM under bf16 autocast)
    logger.info("=" * 60); logger.info(f"Loading model: {MODEL_NAME}"); logger.info("=" * 60)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=False, torch_dtype=torch.float32)
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # verify the FGM target actually exists (fail fast if name mismatch)
    if USE_FGM:
        matched = [n for n, p in model.named_parameters()
                   if p.requires_grad and args.emb_name in n]
        logger.info(f"FGM will perturb {len(matched)} param(s): {matched}")
        if not matched:
            logger.error(f"No parameter name contains '{args.emb_name}'. "
                         f"Check --emb_name (try 'embeddings')."); sys.exit(1)

    # 4. training args (bf16, otherwise identical to baseline) ---------------
    logger.info("=" * 60); logger.info("Configuring training"); logger.info("=" * 60)
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
        logging_dir=str(PROJECT_ROOT / "logs" / "tensorboard" / "deberta_task_a_fgm"),
        report_to=["tensorboard"],
        seed=SEED,
        bf16=True, fp16=False,
        dataloader_num_workers=4,
        remove_unused_columns=True,
    )

    # 5. trainer -------------------------------------------------------------
    trainer = FGMTrainer(
        use_fgm=USE_FGM, epsilon=args.epsilon, emb_name=args.emb_name,
        class_weights=class_weights,
        model=model, args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics_standard,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 6. train (with wall-clock timing) -------------------------------------
    logger.info("=" * 60); logger.info("Starting training"); logger.info("=" * 60)
    t0 = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - t0
    logger.info("Training complete.")
    logger.info(f"  Final train loss : {train_result.training_loss:.4f}")
    logger.info(f"  Best checkpoint  : {trainer.state.best_model_checkpoint}")
    logger.info(f"  Best F1 (dev)    : {trainer.state.best_metric:.4f}")
    logger.info(f"  Total train time : {train_seconds:.1f}s ({train_seconds/60:.1f} min)")
    train_metrics = train_result.metrics
    train_metrics["total_train_time_seconds"] = round(train_seconds, 1)
    trainer.save_metrics("train", train_metrics)

    # 7. threshold optimisation on dev (identical to baseline) ---------------
    logger.info("=" * 60); logger.info("Optimizing decision threshold on DEV"); logger.info("=" * 60)
    val_out = trainer.predict(tokenized_datasets["validation"])
    val_probs = torch.nn.functional.softmax(torch.tensor(val_out.predictions), dim=-1)[:, 1].numpy()
    val_labels = val_out.label_ids
    best_f1, best_thresh = 0.0, 0.5
    for th in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(val_labels, (val_probs >= th).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_thresh = f1, th
    f1_at_half = f1_score(val_labels, np.argmax(val_out.predictions, axis=-1), average="macro")
    logger.info(f"Optimal threshold: {best_thresh:.2f}")
    logger.info(f"Dev macro F1 @0.50        : {f1_at_half:.4f}")
    logger.info(f"Dev macro F1 @{best_thresh:.2f} (best) : {best_f1:.4f}")

    # 8. test with optimised threshold --------------------------------------
    logger.info("=" * 60); logger.info("Evaluating on TEST (optimised threshold)"); logger.info("=" * 60)
    test_out = trainer.predict(tokenized_datasets["test"])
    test_probs = torch.nn.functional.softmax(torch.tensor(test_out.predictions), dim=-1)[:, 1].numpy()
    test_labels = test_out.label_ids
    test_results = compute_metrics_with_threshold(test_probs, test_labels, best_thresh)
    logger.info("Test set results (Optimized):")
    for k, v in test_results.items():
        logger.info(f"  {k:25s} : {v:.4f}" if isinstance(v, float) else f"  {k:25s} : {v}")
    logger.info("\n" + classification_report(
        test_labels, (test_probs >= best_thresh).astype(int),
        target_names=["not_sexist", "sexist"], digits=4))
    trainer.save_metrics("test", test_results)

    # 9. save ----------------------------------------------------------------
    logger.info("=" * 60); logger.info("Saving final model"); logger.info("=" * 60)
    trainer.save_model(MODEL_DIR / "final")
    tokenizer.save_pretrained(MODEL_DIR / "final")
    summary = {
        "model": MODEL_NAME, "task": "Task A - Binary Sexism Detection",
        "script": "train_deberta_fgm.py", "fgm_enabled": USE_FGM,
        "fgm_epsilon": args.epsilon, "fgm_emb_name": args.emb_name,
        "precision": "bf16", "seed": SEED, "max_length": MAX_LENGTH,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr, "weight_decay": 0.01,
        "train_samples": len(dataset["train"]), "dev_samples": len(dataset["validation"]),
        "test_samples": len(dataset["test"]),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_f1_macro_0.5": float(trainer.state.best_metric) if trainer.state.best_metric else None,
        "best_dev_f1_macro_tuned": float(best_f1), "optimal_threshold": float(best_thresh),
        "test_f1_macro": float(test_results.get("f1_macro", 0)),
        "test_f1_sexist": float(test_results.get("f1_sexist", 0)),
        "test_f1_not_sexist": float(test_results.get("f1_not_sexist", 0)),
        "test_accuracy": float(test_results.get("accuracy", 0)),
        "total_train_time_seconds": round(train_seconds, 1),
        "timestamp": timestamp,
    }
    with open(MODEL_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {MODEL_DIR / 'training_summary.json'}")
    logger.info("=" * 60); logger.info("FGM training complete!"); logger.info("=" * 60)
