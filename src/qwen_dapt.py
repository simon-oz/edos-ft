#!/usr/bin/env python3
"""
qwen_dapt.py
Domain-Adaptive Pre-Training (DAPT) / continued pre-training for Qwen2.5-14B-Instruct
on UNLABELED EDOS text, using the causal-LM (next-token) objective.

Difference from DeBERTa DAPT: DeBERTa (encoder) used MLM; Qwen (decoder) uses
next-token prediction. In both cases LABELS ARE IGNORED — only the `text` column
is consumed.

Approach:
  * Load Qwen2.5-14B-Instruct (bf16) + apply a GENTLE LoRA so we adapt the domain
    knowledge without catastrophic forgetting of instruction-following.
  * Train next-token-prediction loss over ALL tokens of the in-domain text.
  * Merge the trained LoRA into the base and save the merged model, so the existing
    SFT scripts can use it as a better base via --model_name (mirrors how DeBERTa
    SFT loads models/deberta/dapt_mlm/final).

Input (unlabeled text; labels ignored):
  data/processed/task_a_train.csv                       (default)
  + optionally task_a_dev.csv / task_a_test.csv via --include_dev_test

Output:
  models/qwen/dapt_lm/final_adapter/   -> the DAPT LoRA adapter (backup)
  models/qwen/dapt_lm/final/           -> merged base + DAPT (loadable by SFT scripts)
  logs/qwen_dapt_*.log

Usage:
  CUDA_VISIBLE_DEVICES=0 python src/qwen_dapt.py
  CUDA_VISIBLE_DEVICES=0 python src/qwen_dapt.py --epochs 2 --lr 5e-5 --include_dev_test
  # then use it for SFT:
  python src/train_qwen_task_a.py --model_name models/qwen/dapt_lm/final --use_16bit_lora

Note: this uses 16-bit LoRA (not QLoRA) on purpose — a 4-bit quantized base cannot be
merged back cleanly, and we need a clean merge to produce the adapted base.
"""
import sys, logging, argparse, random
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, EarlyStoppingCallback, set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

# ------------------------------------------------------------------ paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "qwen" / "dapt_lm"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"qwen_dapt_{timestamp}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")


# ------------------------------------------------------------------ collator
class DAPTCollator:
    """Causal-LM collator: labels = input_ids, with pad positions masked to -100
    so the next-token loss is computed only over real tokens."""
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [b["text"] for b in batch]
        tok = self.tokenizer(texts, max_length=self.max_length, padding=True,
                             truncation=True, return_tensors="pt")
        input_ids = tok["input_ids"]
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        tok["labels"] = labels
        return tok


def load_texts(args):
    """Read ONLY the text column (labels ignored). Default: train split."""
    paths = [PROC_DIR / "task_a_train.csv"]
    if args.include_dev_test:
        paths += [PROC_DIR / "task_a_dev.csv", PROC_DIR / "task_a_test.csv"]
    texts = []
    for p in paths:
        if not p.exists():
            logger.error(f"Missing: {p}"); sys.exit(1)
        df = pd.read_csv(p)
        texts.extend(df["text"].astype(str).tolist())
    logger.info(f"Collected {len(texts)} unlabeled texts from {[str(p.name) for p in paths]}")
    return texts


def parse_args():
    p = argparse.ArgumentParser(description="DAPT (continued pre-training) for Qwen on unlabeled EDOS text")
    p.add_argument("--model_name", type=str, default="/data/models/qwen/qwen2.5-14b-it")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include_dev_test", action="store_true",
                   help="Also use dev/test text (no labels are used, but train-only is the safer default)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    logger.info("=" * 60); logger.info("Qwen DAPT (continued pre-training, next-token)"); logger.info("=" * 60)
    logger.info(f"Model : {args.model_name}")
    logger.info(f"lr={args.lr} epochs={args.epochs} lora r/alpha={args.lora_r}/{args.lora_alpha} "
                f"eff_batch={args.batch_size*args.grad_accum}")

    # 1. unlabeled text -------------------------------------------------------------
    texts = load_texts(args)
    random.seed(args.seed); random.shuffle(texts)
    n_dev = min(500, max(1, len(texts) // 20))          # small held-out slice for LM-loss monitoring
    dapt_dev_texts, train_texts = texts[:n_dev], texts[n_dev:]
    logger.info(f"Train texts: {len(train_texts)}  |  DAPT-dev (LM-loss monitor): {len(dapt_dev_texts)}")

    # 2. tokenizer ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"                     # standard for CLM training
    # mark document boundaries so the model learns where each post ends
    train_texts = [t.strip() + tokenizer.eos_token for t in train_texts]
    dapt_dev_texts = [t.strip() + tokenizer.eos_token for t in dapt_dev_texts]

    train_dataset = Dataset.from_dict({"text": train_texts})
    dev_dataset = Dataset.from_dict({"text": dapt_dev_texts})

    # 3. model + gentle LoRA --------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM))
    model.print_trainable_parameters()

    # 4. training args (monitor LM loss on the held-out slice) ----------------------
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        weight_decay=0.01, warmup_ratio=0.05, logging_steps=20,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        bf16=True, fp16=False, dataloader_num_workers=4, remove_unused_columns=False,
        report_to=["tensorboard"], logging_dir=str(PROJECT_ROOT / "logs" / "tensorboard" / "qwen_dapt"),
        seed=args.seed)

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_dataset, eval_dataset=dev_dataset,
                      processing_class=tokenizer, data_collator=DAPTCollator(tokenizer, args.max_length),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])

    # 5. train ----------------------------------------------------------------------
    logger.info("Starting DAPT...")
    res = trainer.train()
    logger.info(f"DAPT done. train_loss={res.training_loss:.4f}")

    # 6. save adapter, then merge into base and save the adapted base ---------------
    model.save_pretrained(MODEL_DIR / "final_adapter")
    tokenizer.save_pretrained(MODEL_DIR / "final_adapter")
    logger.info(f"Saved DAPT adapter -> {MODEL_DIR / 'final_adapter'}")

    logger.info("Merging LoRA into base (this produces the adapted base for SFT)...")
    model = model.merge_and_unload()
    model.save_pretrained(MODEL_DIR / "final")
    tokenizer.save_pretrained(MODEL_DIR / "final")
    logger.info(f"Saved merged DAPT base -> {MODEL_DIR / 'final'}")
    logger.info("Qwen DAPT complete!")


if __name__ == "__main__":
    main()