#!/usr/bin/env python3
"""
train_deberta_dapt.py
Domain-Adaptive Pre-Training (DAPT) for DeBERTa-v3-large.

Continues pre-training DeBERTa using Masked Language Modeling (MLM) 
on the unlabeled Gab/Reddit corpus. This adapts the model to the 
domain's slang and style before fine-tuning for classification.

Input:
  data/processed/unlabeled_combined_400k.csv

Output:
  models/deberta/dapt_mlm/                 — The domain-adapted base model

Usage:
  CUDA_VISIBLE_DEVICES=0 python src/train_deberta_dapt.py \
      --model_path /data/models/microsoft/deberta-v3-large \
      --batch_size 8 \
      --grad_accum 4 \
      --epochs 1 \
      --lr 1e-5
"""

import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)

# ------------------------------------------------------------------
# Auto-detect project root
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "deberta" / "dapt_mlm"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Logging setup: console + file
# ------------------------------------------------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_deberta_dapt_{timestamp}.log"

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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    return parser.parse_args()

def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

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
    # 1. Load Unlabeled Data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Loading Unlabeled Datasets for DAPT")
    logger.info("=" * 60)

    unlabeled_path = PROC_DIR / "unlabeled_combined_400k.csv"
    if not unlabeled_path.exists():
        logger.error(f"Missing: {unlabeled_path}. Run parse_edos_data.py first.")
        sys.exit(1)

    dataset = load_dataset("csv", data_files={"train": str(unlabeled_path)})
    
    # Drop any empty texts just in case
    dataset = dataset.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)
    
    logger.info(f"Train      : {len(dataset['train'])} samples")

    # ------------------------------------------------------------------
    # 2. Tokenize
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH), 
        batched=True,
        remove_columns=["text"] # Remove text column, keep only input_ids/attention_mask
    )

    # ------------------------------------------------------------------
    # 3. Data Collator for Masked Language Modeling
    # ------------------------------------------------------------------
    # This dynamically masks 15% of tokens in the batch, generating labels for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob
    )

    # ------------------------------------------------------------------
    # 4. Load model
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Loading model for MLM: {MODEL_NAME}")
    logger.info("=" * 60)

    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32, # Master weights in FP32 for stable AMP
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters     : {total_params:,}")

    # ------------------------------------------------------------------
    # 5. Training arguments
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Configuring DAPT training")
    logger.info("=" * 60)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        eval_strategy="no",         # No eval set for unsupervised DAPT
        save_strategy="epoch",
        learning_rate=args.lr,                           
        per_device_train_batch_size=args.batch_size,     
        gradient_accumulation_steps=args.grad_accum,     
        num_train_epochs=args.epochs,                    
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        report_to=["tensorboard"],
        seed=SEED,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False, # Must be False for DataCollatorForLanguageModeling
    )

    logger.info(f"Model path           : {MODEL_NAME}")
    logger.info(f"Learning rate        : {training_args.learning_rate}")
    logger.info(f"Train batch size     : {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"Max epochs           : {training_args.num_train_epochs}")
    logger.info(f"MLM Probability      : {args.mlm_prob}")

    # ------------------------------------------------------------------
    # 6. Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting DAPT training")
    logger.info("=" * 60)

    train_result = trainer.train()

    logger.info("DAPT Training complete.")
    logger.info(f"  Final train loss : {train_result.training_loss:.4f}")

    # ------------------------------------------------------------------
    # 8. Save model and tokenizer
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Saving domain-adapted model")
    logger.info("=" * 60)

    trainer.save_model(MODEL_DIR / "final")
    tokenizer.save_pretrained(MODEL_DIR / "final")

    logger.info(f"Saved DAPT model to: {MODEL_DIR / 'final'}")
    logger.info("=" * 60)
    logger.info("DAPT complete! Use this model for downstream fine-tuning.")
    logger.info("=" * 60)