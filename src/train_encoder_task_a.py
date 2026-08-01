#!/usr/bin/env python3
"""
train_encoder_task_a.py  (generalized encoder trainer + probability emitter)
Trains ANY HuggingFace sequence-classification encoder on EDOS Task A with the
proven recipe (class-weighted CE, bf16, early-stop on f1_macro) and WRITES the
model's dev/test P(sexist) probability vectors to a shared folder so that
ensemble_vote.py can stack heterogeneous members (DeBERTa + twHIN + ...).

This is the keystone of the diversity ensemble: each member must emit probs with
the SAME row order as the CSVs (it does -- we never shuffle at predict time).

Robustness baked in (lessons from prior debugging):
  * label->labels rename + labels copied in preprocess + remove only 'text'
    => the loss can NEVER see None labels (the bug that crashed us repeatedly).
  * bf16 default (no GradScaler => no "unscale FP16 gradients" crash).
  * compute_loss accepts **kwargs (transformers v5.x passes num_items_in_batch).
  * metric_for_best_model="f1_macro" exactly matches compute_metrics output.
  * collapse is WARNED, never exit() (exit() killed runs mid-training before).

Usage (run each member once; point --model_path at the backbone you want):
  # Member 1: DeBERTa fine-tuned from the DAPT checkpoint (your 0.8593 recipe)
  CUDA_VISIBLE_DEVICES=0 python src/train_encoder_task_a.py \
      --model_path models/deberta/dapt_mlm/final --out_tag deberta_dapt

  # Member 2: twHIN-BERT-large (social-media pretrained; the SOTA's partner model)
  CUDA_VISIBLE_DEVICES=0 python src/train_encoder_task_a.py \
      --model_path /data/models/Twitter/twhin-bert-large  --out_tag twhin
  # Note, using the following cmd to donwload the model
  # hf download Twitter/twhin-bert-large --local-dir /data/models/Twitter

  # (H100 headroom) bigger batch is free and faster:
  ... --batch_size 32
"""
import sys, gc, json, logging, argparse, time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback, set_seed,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
PROC_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_PROBS_DIR = PROJECT_ROOT / "models" / "ensemble_probs"

LOG_DIR = PROJECT_ROOT / "logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_encoder_task_a_{timestamp}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")


# ------------------------------------------------------------------
# Class-weighted loss (the recipe behind 0.8593)
# ------------------------------------------------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *a, **kw):
        super().__init__(*a, **kw); self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # **kwargs for v5.x
        labels = inputs.pop("labels")
        outputs = model(**inputs); logits = outputs.logits
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.model.config.num_labels), labels.view(-1), weight=w)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    n_pos, n_neg = int(preds.sum()), int((1 - preds).sum())
    if n_pos == 0 or n_neg == 0:
        logger.warning(f"  !! COLLAPSE at eval: not_sexist={n_neg}, sexist={n_pos}")
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    pc = f1_score(labels, preds, average=None, zero_division=0)
    return {"accuracy": float((preds == labels).mean()), "f1_macro": float(f1),
            "f1_not_sexist": float(pc[0]) if len(pc) > 0 else 0.0,
            "f1_sexist": float(pc[1]) if len(pc) > 1 else 0.0,
            "n_pred_sexist": n_pos, "n_pred_not_sexist": n_neg}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="HF name or local path of any seq-cls encoder (DeBERTa, twHIN, ...)")
    p.add_argument("--out_tag", required=True,
                   help="Short tag for this member, e.g. deberta_dapt / twhin. "
                        "Probs are saved as {out_tag}_dev_probs.npy etc.")
    p.add_argument("--probs_dir", type=str, default=str(DEFAULT_PROBS_DIR))
    p.add_argument("--model_save_dir", type=str, default=None,
                   help="Where to save the finetuned model (default: models/encoders/{out_tag})")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_class_weights", dest="use_class_weights", action="store_false")
    p.set_defaults(use_class_weights=True)
    p.add_argument("--fp16", action="store_true", default=False,
                   help="Use fp16 instead of the default bf16 (bf16 is safer on H100/Ada)")
    p.add_argument("--no_save_model", dest="save_model", action="store_false")
    p.set_defaults(save_model=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    probs_dir = Path(args.probs_dir); probs_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir = Path(args.model_save_dir) if args.model_save_dir \
        else (PROJECT_ROOT / "models" / "encoders" / args.out_tag)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        logger.error("CUDA not available."); sys.exit(1)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} "
                f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
    logger.info("=" * 60)
    logger.info(f"MEMBER tag      : {args.out_tag}")
    logger.info(f"Model path      : {args.model_path}")
    logger.info(f"Precision       : {'fp16' if args.fp16 else 'bf16'}")
    logger.info(f"Class weights   : {args.use_class_weights}")
    logger.info(f"Batch / accum   : {args.batch_size} / {args.grad_accum} "
                f"(eff {args.batch_size*args.grad_accum})  lr={args.lr}  epochs={args.epochs}")
    logger.info(f"Probs out dir   : {probs_dir}")
    logger.info("=" * 60)

    # ---- 1. data ----
    ds = load_dataset("csv", data_files={
        "train": str(PROC_DIR / "task_a_train.csv"),
        "validation": str(PROC_DIR / "task_a_dev.csv"),
        "test": str(PROC_DIR / "task_a_test.csv")})
    ds = ds.rename_column("label", "labels")          # canonical name -> no internal ambiguity
    train_labels = np.array(ds["train"]["labels"], dtype=np.int64)
    n1, n0 = int(train_labels.sum()), int((train_labels == 0).sum()); n = len(train_labels)
    logger.info(f"Train={n} (sexist={n1}, not={n0})  Dev={len(ds['validation'])}  "
                f"Test={len(ds['test'])}")

    # ---- 2. tokenizer + canonical preprocess (labels copied back; only 'text' dropped) ----
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
        logger.info(f"pad_token was None -> set to {tok.pad_token}")

    def preprocess(ex):
        r = tok(ex["text"], truncation=True, padding=False, max_length=args.max_length)
        r["labels"] = ex["labels"]                    # keep labels through the map
        return r
    orig_cols = ds["train"].column_names
    ds = ds.map(preprocess, batched=True, remove_columns=orig_cols)
    collator = DataCollatorWithPadding(tokenizer=tok, padding="longest")

    # ---- 3. model ----
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=2, ignore_mismatched_sizes=False, torch_dtype=torch.float32)
    logger.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    cw = None
    if args.use_class_weights:
        cw = torch.tensor([n / (2 * n0), n / (2 * n1)], dtype=torch.float)
        logger.info(f"Class weights: [not={cw[0]:.3f}, sexist={cw[1]:.3f}]")

    # ---- 4. training args ----
    steps_per_ep = max(1, n // (args.batch_size * args.grad_accum))
    total_steps = steps_per_ep * args.epochs
    warmup_steps = int(0.1 * total_steps)
    ta = TrainingArguments(
        output_dir=str(model_save_dir), eval_strategy="epoch", save_strategy="epoch",
        learning_rate=args.lr, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32, gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, weight_decay=0.01, warmup_steps=warmup_steps,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        logging_steps=50, seed=args.seed, fp16=args.fp16, bf16=not args.fp16,
        dataloader_num_workers=4, remove_unused_columns=True, report_to="none",
        save_total_limit=2)

    trainer = WeightedTrainer(
        class_weights=cw, model=model, args=ta,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        processing_class=tok, data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

    # ---- 5. train ----
    logger.info("Starting training..."); t0 = time.time()
    try:
        res = trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("OOM: lower --batch_size or raise --grad_accum."); raise
        raise
    secs = time.time() - t0
    logger.info(f"Training done in {secs:.0f}s. best dev f1_macro = "
                f"{trainer.state.best_metric:.4f} @ {trainer.state.best_model_checkpoint}")

    # ---- 6. EMIT probability vectors (the contract with ensemble_vote.py) ----
    def pos_probs(split_ds):
        out = trainer.predict(split_ds)
        return torch.nn.functional.softmax(torch.tensor(out.predictions), dim=-1)[:, 1].numpy()

    dev_probs = pos_probs(ds["validation"])
    test_probs = pos_probs(ds["test"])
    dev_p = probs_dir / f"{args.out_tag}_dev_probs.npy"
    test_p = probs_dir / f"{args.out_tag}_test_probs.npy"
    np.save(dev_p, dev_probs); np.save(test_p, test_probs)
    logger.info("=" * 60)
    logger.info(f"SAVED {dev_p}   shape={dev_probs.shape}")
    logger.info(f"SAVED {test_p}  shape={test_probs.shape}")
    logger.info("(row order == CSV order; ensemble_vote.py will glob these)")
    logger.info("=" * 60)

    # ---- 7. single-model reference (tuned threshold on dev) + save ----
    dev_labels = np.array(ds["validation"]["labels"]); test_labels = np.array(ds["test"]["labels"])
    bt, bf = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(dev_labels, (dev_probs >= t).astype(int), average="macro", zero_division=0)
        if f > bf: bf, bt = f, t
    test_m = compute_metrics.__wrapped__ if False else None  # placeholder to keep linters quiet
    tp = (test_probs >= bt).astype(int)
    pc = f1_score(test_labels, tp, average=None, zero_division=0)
    logger.info(f"[single-model ref] dev f1_macro@{bt:.2f}={bf:.4f} | "
                f"test f1_macro={f1_score(test_labels, tp, average='macro', zero_division=0):.4f} "
                f"(sexist={pc[1]:.4f}, not={pc[0]:.4f})  -- reference only; trust the voter")
    if args.save_model:
        trainer.save_model(str(model_save_dir / "final")); tok.save_pretrained(str(model_save_dir / "final"))
    summary = {"out_tag": args.out_tag, "model_path": args.model_path,
               "precision": "fp16" if args.fp16 else "bf16", "lr": args.lr,
               "use_class_weights": args.use_class_weights, "seed": args.seed,
               "best_dev_f1_macro": float(trainer.state.best_metric) if trainer.state.best_metric else None,
               "single_ref_test_f1_macro": float(f1_score(test_labels, tp, average="macro", zero_division=0)),
               "train_seconds": round(secs, 1), "timestamp": timestamp}
    json.dump(summary, open(model_save_dir / "member_summary.json", "w"), indent=2)
    logger.info(f"Member summary -> {model_save_dir/'member_summary.json'}")
    logger.info(f"Member '{args.out_tag}' complete. Run more members, then ensemble_vote.py.")