#!/usr/bin/env python3
"""
train_qwen_task_a.py
Fine-tune Qwen2.5-14B-Instruct (or Qwen2.6-13B) on EDOS Task A (binary sexism detection).

Supports three modes:
  1. QLoRA 4-bit  (default, ~35 GB VRAM, safest)
  2. LoRA 16-bit  (better quality, ~55 GB VRAM)
  3. Full fine-tuning with DeepSpeed ZeRO-2 + CPU offload (~65 GB VRAM)

Input:
  data/processed/task_a_{train,dev}.csv

Output:
  models/qwen/task_a/                      — adapters or full model checkpoints
  logs/train_qwen_task_a_*.log             — training log

Usage:
  # Default: QLoRA 4-bit (recommended start)
  python src/train_qwen_task_a.py

  # LoRA 16-bit
  python src/train_qwen_task_a.py --use_16bit_lora

  # Full fine-tuning with DeepSpeed
  python src/train_qwen_task_a.py --full_finetune --deepspeed_config src/ds_config_zero2.json

  # Different model size
  python src/train_qwen_task_a.py --model_name Qwen/Qwen2.5-14B-Instruct
"""

import sys
import json
import re
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, classification_report, accuracy_score
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# ------------------------------------------------------------------
# Auto-detect project root
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

# FIX: Point to the correct directory containing the processed CSVs
PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "qwen" / "task_a"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_qwen_task_a_{timestamp}.log"

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

# ------------------------------------------------------------------
# System prompt with EDOS Task A definition
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert annotator for the Explainable Detection of Online Sexism (EDOS) dataset.

Your task is to classify whether a given text is sexist or not sexist.

Definitions:
- Sexist: The text expresses sexism, including overt hostility, implicit bias, stereotypes, or objectification toward women. This includes sarcasm, dog-whistles, backhanded compliments, and seemingly neutral statements that reinforce gender stereotypes.
- Not Sexist: The text does not express sexism. It may discuss gender-related topics neutrally, or be completely unrelated.

Respond with ONLY a valid JSON object in this exact format:
{"classification": "sexist"}
or
{"classification": "not_sexist"}"""


def format_chat_example(text: str, label: int) -> Dict[str, List[Dict]]:
    """Format a single example into Qwen chat messages."""
    answer = "sexist" if label == 1 else "not_sexist"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Text: "{text}"'},
        {"role": "assistant", "content": json.dumps({"classification": answer})},
    ]
    return {"messages": messages}


def parse_generated_output(text: str) -> int:
    """Parse model output to extract classification. Returns 1=sexist, 0=not_sexist, -1=invalid."""
    # Try to find JSON pattern
    match = re.search(r'"classification"\s*:\s*"(\w+)"', text)
    if match:
        val = match.group(1).lower()
        if val == "sexist":
            return 1
        elif val in ("not_sexist", "not sexist"):
            return 0
    # Fallback: keyword search
    text_lower = text.lower()
    if "not_sexist" in text_lower or "not sexist" in text_lower:
        return 0
    if "sexist" in text_lower and "not" not in text_lower:
        return 1
    return -1  # Invalid / could not parse


# ------------------------------------------------------------------
# Custom data collator with label masking
# ------------------------------------------------------------------
class TaskADataCollator:
    """
    Collates chat-formatted examples and masks non-assistant tokens
    so loss is only computed on the assistant's JSON response.
    """

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Apply chat template to each example
        texts = []
        for item in batch:
            chat_text = self.tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(chat_text)

        # Tokenize all
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"]
        labels = input_ids.clone()

        # Mask non-assistant tokens: find "<|im_start|>assistant\n" and mask everything before it
        assistant_token_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        # Some tokenizers tokenize differently; try variations
        if len(assistant_token_ids) < 2:
            assistant_token_ids = self.tokenizer.encode("assistant", add_special_tokens=False)

        for i in range(len(batch)):
            seq = input_ids[i].tolist()
            # Find the LAST occurrence of assistant marker (the response we want to train on)
            # Search for the assistant token sequence
            found_pos = -1
            for j in range(len(seq) - len(assistant_token_ids), -1, -1):
                if seq[j:j + len(assistant_token_ids)] == assistant_token_ids:
                    found_pos = j
                    break

            if found_pos != -1:
                # Mask everything before and including the assistant marker itself
                labels[i, :found_pos + len(assistant_token_ids)] = -100
            else:
                # Fallback: if we can't find the marker, mask all but last 10 tokens
                # (should not happen with proper chat templates)
                labels[i, :-10] = -100
                logger.warning(f"Could not find assistant marker in batch item {i}")

        tokenized["labels"] = labels
        return tokenized


# ------------------------------------------------------------------
# Metrics for generation-based classification
# ------------------------------------------------------------------
def evaluate_model(model, tokenizer, eval_dataset, device="cuda", batch_size=8, max_new_tokens=50):
    """Run generation-based evaluation and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            texts = batch["text"]
            labels = batch["label"].tolist()

            # Build prompts (without assistant response)
            prompts = []
            for text in texts:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f'Text: "{text}"'},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode only the generated part
            generated = outputs[:, inputs["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for dec, true_label in zip(decoded, labels):
                pred = parse_generated_output(dec)
                if pred == -1:
                    # Default to majority class (not_sexist = 0) if parsing fails
                    pred = 0
                all_preds.append(pred)
                all_labels.append(true_label)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    # Confusion matrix components
    tp = sum((p == 1 and l == 1) for p, l in zip(all_preds, all_labels))
    fp = sum((p == 1 and l == 0) for p, l in zip(all_preds, all_labels))
    fn = sum((p == 0 and l == 1) for p, l in zip(all_preds, all_labels))
    tn = sum((p == 0 and l == 0) for p, l in zip(all_preds, all_labels))

    metrics = {
        "eval_accuracy": acc,
        "eval_f1_macro": f1_macro,
        "eval_precision_not_sexist": precision[0] if len(precision) > 0 else 0,
        "eval_recall_not_sexist": recall[0] if len(recall) > 0 else 0,
        "eval_f1_not_sexist": f1_per_class[0] if len(f1_per_class) > 0 else 0,
        "eval_precision_sexist": precision[1] if len(precision) > 1 else 0,
        "eval_recall_sexist": recall[1] if len(recall) > 1 else 0,
        "eval_f1_sexist": f1_per_class[1] if len(f1_per_class) > 1 else 0,
        "eval_tp": tp,
        "eval_fp": fp,
        "eval_fn": fn,
        "eval_tn": tn,
        "eval_parse_failures": sum(1 for p in all_preds if p == -1),
    }

    return metrics, all_preds, all_labels


# ------------------------------------------------------------------
# Custom Trainer for generation-based evaluation
# ------------------------------------------------------------------
class QwenTrainer(Trainer):
    def __init__(self, gen_eval_dataset=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_eval_dataset = gen_eval_dataset

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Skip standard evaluation and run generation-based evaluation instead
        if self.gen_eval_dataset is None:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            
        self.model.eval()
        
        start_time = time.time()
        metrics, _, _ = evaluate_model(
            self.model, 
            self.processing_class, 
            self.gen_eval_dataset, 
            batch_size=8
        )
        eval_time = time.time() - start_time
        
        # Add runtime metrics just so Trainer logs them cleanly
        metrics["eval_runtime"] = round(eval_time, 4)
        metrics["eval_samples_per_second"] = round(len(self.gen_eval_dataset) / eval_time, 4) if eval_time > 0 else 0.0
        metrics["eval_steps_per_second"] = 0.0
        
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen on EDOS Task A")
    parser.add_argument("--model_name", type=str, default="/data/models/qwen/qwen2.5-14b-it",
                        help="Path to local model or HuggingFace model name")
    parser.add_argument("--use_16bit_lora", action="store_true",
                        help="Use LoRA in 16-bit instead of 4-bit QLoRA")
    parser.add_argument("--full_finetune", action="store_true",
                        help="Full fine-tuning (requires DeepSpeed or very large VRAM)")
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="Path to DeepSpeed config JSON for full fine-tuning")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Max training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (LoRA) or 1e-5 (full FT)")
    parser.add_argument("--lora_r", type=int, default=128,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=256,
                        help="LoRA alpha")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every N steps")
    return parser.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("Qwen Task A Fine-Tuning")
    logger.info("=" * 60)
    logger.info(f"Model              : {args.model_name}")
    logger.info(f"Mode               : {'Full FT' if args.full_finetune else ('16-bit LoRA' if args.use_16bit_lora else '4-bit QLoRA')}")
    logger.info(f"Max length         : {args.max_length}")
    logger.info(f"Batch size         : {args.batch_size}")
    logger.info(f"Gradient accum     : {args.grad_accum}")
    logger.info(f"Effective batch    : {args.batch_size * args.grad_accum}")
    logger.info(f"Epochs             : {args.epochs}")
    logger.info(f"Learning rate      : {args.lr}")
    logger.info(f"LoRA r/alpha       : {args.lora_r}/{args.lora_alpha}")
    logger.info(f"Seed               : {args.seed}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Loading datasets")
    logger.info("=" * 60)

    train_path = PROC_DIR / "task_a_train.csv"
    dev_path = PROC_DIR / "task_a_custom_dev.csv"
    test_path = PROC_DIR / "task_a_test.csv"

    # Fallback to official dev if custom_dev doesn't exist yet
    if not dev_path.exists():
        dev_path = PROC_DIR / "task_a_dev.csv"
        logger.info(f"custom_dev not found, using official dev: {dev_path}")

    for p in [train_path, dev_path, test_path]:
        if not p.exists():
            logger.error(f"Missing: {p}")
            sys.exit(1)

    df_train = pd.read_csv(train_path)
    df_dev = pd.read_csv(dev_path)
    df_test = pd.read_csv(test_path)

    logger.info(f"Train samples : {len(df_train)}")
    logger.info(f"Dev samples   : {len(df_dev)}")
    logger.info(f"Test samples  : {len(df_test)}")
    logger.info(f"Train class distribution: {dict(df_train['label'].value_counts().sort_index())}")

    # Format for causal LM
    train_data = [format_chat_example(row["text"], row["label"]) for _, row in df_train.iterrows()]
    dev_data = [format_chat_example(row["text"], row["label"]) for _, row in df_dev.iterrows()]

    train_dataset = Dataset.from_list(train_data)
    dev_dataset = Dataset.from_list(dev_data)

    # Keep raw text/label for evaluation
    dev_eval = [{"text": row["text"], "label": row["label"]} for _, row in df_dev.iterrows()]
    test_eval = [{"text": row["text"], "label": row["label"]} for _, row in df_test.iterrows()]


    # ------------------------------------------------------------------
    # 2. Load tokenizer
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Loading tokenizer: {args.model_name}")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for generation

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Loading model")
    logger.info("=" * 60)

    if args.full_finetune:
        # Full fine-tuning: load in bfloat16, no quantization
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.gradient_checkpointing_enable()
        logger.info("Model loaded in bfloat16 for full fine-tuning")

    elif args.use_16bit_lora:
        # LoRA 16-bit: load in bfloat16, no quantization
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("Model loaded in bfloat16 for 16-bit LoRA")

    else:
        # Default: QLoRA 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        logger.info("Model loaded in 4-bit NF4 for QLoRA")

    # ------------------------------------------------------------------
    # 4. Apply LoRA (if not full fine-tuning)
    # ------------------------------------------------------------------
    if not args.full_finetune:
        logger.info("=" * 60)
        logger.info("Applying LoRA")
        logger.info("=" * 60)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info(f"LoRA applied: r={args.lora_r}, alpha={args.lora_alpha}")

    # ------------------------------------------------------------------
    # 5. Data collator
    # ------------------------------------------------------------------
    data_collator = TaskADataCollator(tokenizer, max_length=args.max_length)

    # ------------------------------------------------------------------
    # 6. Training arguments
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Configuring training")
    logger.info("=" * 60)

    effective_batch = args.batch_size * args.grad_accum
    lr = args.lr if not args.full_finetune else 1e-5
    wd = 0.01 if not args.full_finetune else 0.1

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        weight_decay=wd,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        bf16=True,
        fp16=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        logging_dir=str(PROJECT_ROOT / "logs" / "tensorboard" / "qwen_task_a"),
        seed=args.seed,
        deepspeed=args.deepspeed_config if args.full_finetune else None,
    )

    logger.info(f"Learning rate      : {lr}")
    logger.info(f"Weight decay       : {wd}")
    logger.info(f"Effective batch    : {effective_batch}")
    logger.info(f"Max epochs         : {args.epochs}")
    logger.info(f"Eval every         : {args.eval_steps} steps")
    logger.info(f"Save every         : {args.save_steps} steps")

    # ------------------------------------------------------------------
    # 7. Trainer
    # ------------------------------------------------------------------
    # Note: We use our custom QwenTrainer so that evaluate() runs generation
    # and returns eval_f1_macro natively. This allows EarlyStoppingCallback 
    # and load_best_model_at_end to work perfectly.

    trainer = QwenTrainer(
        gen_eval_dataset=dev_eval,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,  # Now passing actual dev_dataset
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ------------------------------------------------------------------
    # 8. Train
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)

    train_result = trainer.train()
    logger.info("Training complete.")
    logger.info(f"  Final train loss: {train_result.training_loss:.4f}")

    # ------------------------------------------------------------------
    # 9. Final evaluation on dev set
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Final evaluation on DEV set (generation-based)")
    logger.info("=" * 60)

    dev_metrics, dev_preds, dev_labels = evaluate_model(
        model, tokenizer, dev_eval, batch_size=8
    )

    for k, v in dev_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:30s}: {v:.4f}")
        else:
            logger.info(f"  {k:30s}: {v}")

    # ------------------------------------------------------------------
    # 9.1 Final evaluation on test set
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Final evaluation on TEST set (generation-based)")
    logger.info("=" * 60)

    test_metrics, test_preds, test_labels = evaluate_model(
        model, tokenizer, test_eval, batch_size=8
    )

    for k, v in test_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:30s}: {v:.4f}")
        else:
            logger.info(f"  {k:30s}: {v}")

    # ------------------------------------------------------------------
    # 10. Save model
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Saving model")
    logger.info("=" * 60)

    if args.full_finetune:
        trainer.save_model(MODEL_DIR / "final")
    else:
        # Save only LoRA adapter
        model.save_pretrained(MODEL_DIR / "final_adapter")
        tokenizer.save_pretrained(MODEL_DIR / "final_adapter")

    tokenizer.save_pretrained(MODEL_DIR / "final")

    # Save summary
    summary = {
        "model": args.model_name,
        "task": "Task A - Binary Sexism Detection",
        "mode": "full_ft" if args.full_finetune else ("16bit_lora" if args.use_16bit_lora else "4bit_qlora"),
        "seed": args.seed,
        "max_length": args.max_length,
        "lora_r": args.lora_r if not args.full_finetune else None,
        "lora_alpha": args.lora_alpha if not args.full_finetune else None,
        "train_samples": len(df_train),
        "dev_samples": len(df_dev),
        "test_samples": len(df_test),
        "final_dev_f1_macro": float(dev_metrics["dev_f1_macro"]),
        "final_dev_accuracy": float(dev_metrics["devaccuracy"]),
        "final_dev_f1_sexist": float(dev_metrics["dev_f1_sexist"]),
        "final_dev_f1_not_sexist": float(dev_metrics["dev_f1_not_sexist"]),
        "final_test_f1_macro": float(test_metrics["test_f1_macro"]),
        "final_test_accuracy": float(test_metrics["test_accuracy"]),
        "final_test_f1_sexist": float(test_metrics["test_f1_sexist"]),
        "final_test_f1_not_sexist": float(test_metrics["test_f1_not_sexist"]),
        "timestamp": timestamp,
    }

    with open(MODEL_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {MODEL_DIR / 'training_summary.json'}")
    logger.info("=" * 60)
    logger.info("Qwen Task A training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()