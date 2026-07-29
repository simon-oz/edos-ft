#!/usr/bin/env python3
"""
error_mining.py
Extract high-confidence misclassifications from a trained DeBERTa model
to build a "hard negative" dataset for augmentation and Qwen2.5-14B training.

Input:
  models/deberta/task_a_baseline/          — trained checkpoint
  data/processed/task_a_train.csv          — training data to analyze

Output:
  data/processed/error_analysis/
    ├── false_negatives_high_conf.csv      — sexist text predicted as not sexist (p > 0.9)
    ├── false_positives_high_conf.csv      — not sexist text predicted as sexist (p > 0.9)
    ├── false_negatives_medium_conf.csv    — sexist text predicted as not sexist (0.7 < p <= 0.9)
    ├── false_positives_medium_conf.csv    — not sexist text predicted as sexist (0.7 < p <= 0.9)
    ├── all_errors.csv                     — all misclassifications with confidence
    └── error_taxonomy_template.json       — template for manual categorization

Usage:
  From project root:
    python src/error_mining.py \
        --checkpoint_dir models/deberta/task_a_baseline/checkpoint-XXXX \
        --output_dir data/processed/error_analysis

  If --checkpoint_dir is omitted, uses the best checkpoint from training_summary.json.
"""

import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------------
# Auto-detect project root
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "deberta" / "task_a_baseline"
DEFAULT_OUTPUT_DIR = PROC_DIR / "error_analysis"

# ------------------------------------------------------------------
# Logging setup: console + file
# ------------------------------------------------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"error_mining_{timestamp}.log"

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

# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Mine errors from trained DeBERTa baseline.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to a specific checkpoint. If None, uses best checkpoint from training_summary.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to save error analysis outputs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    parser.add_argument(
        "--high_conf_threshold",
        type=float,
        default=0.9,
        help="Probability threshold for high-confidence errors",
    )
    parser.add_argument(
        "--medium_conf_threshold",
        type=float,
        default=0.7,
        help="Lower bound for medium-confidence errors",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Resolve checkpoint
    # ------------------------------------------------------------------
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        # Try to read best checkpoint from training summary
        summary_path = MODEL_DIR / "training_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            best_ckpt = summary.get("best_checkpoint")
            if best_ckpt:
                checkpoint_dir = Path(best_ckpt)
                logger.info(f"Using best checkpoint from summary: {checkpoint_dir}")
            else:
                logger.error("No best_checkpoint found in training_summary.json")
                sys.exit(1)
        else:
            # Fallback: find the latest checkpoint directory
            ckpts = sorted(MODEL_DIR.glob("checkpoint-*"), key=lambda p: p.name)
            if not ckpts:
                logger.error(f"No checkpoints found in {MODEL_DIR}")
                sys.exit(1)
            checkpoint_dir = ckpts[-1]
            logger.info(f"Using latest checkpoint: {checkpoint_dir}")

    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    logger.info(f"Checkpoint: {checkpoint_dir.absolute()}")

    # ------------------------------------------------------------------
    # 2. Load model and tokenizer
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Loading model and tokenizer")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.cuda()
    model.eval()

    logger.info(f"Model loaded: {checkpoint_dir.name}")
    logger.info(f"Device: {next(model.parameters()).device}")

    # ------------------------------------------------------------------
    # 3. Load training data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Loading training data for error analysis")
    logger.info("=" * 60)

    train_path = PROC_DIR / "task_a_train.csv"
    if not train_path.exists():
        logger.error(f"Missing: {train_path}")
        sys.exit(1)

    df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(df)} training samples")

    # ------------------------------------------------------------------
    # 4. Run inference
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Running inference on training set")
    logger.info("=" * 60)

    all_preds = []
    all_probs = []
    texts = df["text"].astype(str).tolist()

    batch_size = args.batch_size
    num_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Inference"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            ).to("cuda")

            outputs = model(**inputs)
            batch_logits = outputs.logits
            batch_probs = torch.softmax(batch_logits, dim=-1)
            batch_preds = torch.argmax(batch_probs, dim=-1)

            all_preds.extend(batch_preds.cpu().numpy())
            all_probs.extend(batch_probs.cpu().numpy())

    df["pred"] = all_preds
    df["prob_not_sexist"] = [p[0] for p in all_probs]
    df["prob_sexist"] = [p[1] for p in all_probs]
    df["confidence"] = np.max(all_probs, axis=1)
    df["correct"] = (df["pred"] == df["label"]).astype(int)

    accuracy = df["correct"].mean()
    logger.info(f"Train-set accuracy: {accuracy:.4f}")
    logger.info(f"Total errors      : {(df['correct'] == 0).sum()}")

    # ------------------------------------------------------------------
    # 5. Categorize errors by confidence
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Categorizing errors by confidence")
    logger.info("=" * 60)

    high_thresh = args.high_conf_threshold
    med_thresh = args.medium_conf_threshold

    # False negatives: true=sexist (1), pred=not_sexist (0)
    false_negatives = df[(df["label"] == 1) & (df["pred"] == 0)].copy()
    fn_high = false_negatives[false_negatives["confidence"] >= high_thresh]
    fn_medium = false_negatives[
        (false_negatives["confidence"] >= med_thresh) &
        (false_negatives["confidence"] < high_thresh)
    ]

    # False positives: true=not_sexist (0), pred=sexist (1)
    false_positives = df[(df["label"] == 0) & (df["pred"] == 1)].copy()
    fp_high = false_positives[false_positives["confidence"] >= high_thresh]
    fp_medium = false_positives[
        (false_positives["confidence"] >= med_thresh) &
        (false_positives["confidence"] < high_thresh)
    ]

    logger.info(f"False negatives (total)       : {len(false_negatives)}")
    logger.info(f"  High conf (≥{high_thresh:.1f})   : {len(fn_high)}")
    logger.info(f"  Medium conf ({med_thresh:.1f}–{high_thresh:.1f}): {len(fn_medium)}")
    logger.info(f"False positives (total)       : {len(false_positives)}")
    logger.info(f"  High conf (≥{high_thresh:.1f})   : {len(fp_high)}")
    logger.info(f"  Medium conf ({med_thresh:.1f}–{high_thresh:.1f}): {len(fp_medium)}")

    # ------------------------------------------------------------------
    # 6. Save error files
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Saving error analysis files")
    logger.info("=" * 60)

    # High confidence
    fn_high.to_csv(output_dir / "false_negatives_high_conf.csv", index=False)
    fp_high.to_csv(output_dir / "false_positives_high_conf.csv", index=False)
    logger.info(f"Saved false_negatives_high_conf.csv  : {len(fn_high)} rows")
    logger.info(f"Saved false_positives_high_conf.csv  : {len(fp_high)} rows")

    # Medium confidence
    fn_medium.to_csv(output_dir / "false_negatives_medium_conf.csv", index=False)
    fp_medium.to_csv(output_dir / "false_positives_medium_conf.csv", index=False)
    logger.info(f"Saved false_negatives_medium_conf.csv: {len(fn_medium)} rows")
    logger.info(f"Saved false_positives_medium_conf.csv: {len(fp_medium)} rows")

    # All errors combined
    all_errors = df[df["correct"] == 0].copy()
    all_errors.to_csv(output_dir / "all_errors.csv", index=False)
    logger.info(f"Saved all_errors.csv                 : {len(all_errors)} rows")

    # ------------------------------------------------------------------
    # 7. Error taxonomy template
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Generating error taxonomy template")
    logger.info("=" * 60)

    taxonomy = {
        "description": (
            "Manual categorization of high-confidence errors. "
            "Read ~50 false negatives and ~50 false positives, then categorize."
        ),
        "categories": {
            "sarcasm_irony": {
                "description": "Text uses sarcasm or irony to express sexism",
                "examples": [],
                "count_fn": 0,
                "count_fp": 0,
            },
            "implicit_bias": {
                "description": "Seemingly neutral text that reinforces gender stereotypes",
                "examples": [],
                "count_fn": 0,
                "count_fp": 0,
            },
            "context_dependent": {
                "description": "Requires external world knowledge or thread context",
                "examples": [],
                "count_fn": 0,
                "count_fp": 0,
            },
            "borderline_annotator_disagreement": {
                "description": "Genuinely ambiguous; even humans disagree",
                "examples": [],
                "count_fn": 0,
                "count_fp": 0,
            },
            "slang_abbreviations": {
                "description": "Heavy use of slang, abbreviations, or emojis obscures intent",
                "examples": [],
                "count_fn": 0,
                "count_fp": 0,
            },
            "dog_whistle": {
                "description": "Coded language understood by in-group but not obvious",
                "examples": [],
                "count_fn": 0,
                "count_fp": 0,
            },
            "other": {
                "description": "Does not fit above categories",
                "examples": [],
                "count_fn": 0,
                "count_fp": 0,
            },
        },
        "instructions": (
            "1. Open false_negatives_high_conf.csv and false_positives_high_conf.csv\n"
            "2. Read the first 50 rows of each\n"
            "3. For each error, assign a category key from 'categories' above\n"
            "4. Update the count_fn / count_fp fields\n"
            "5. Add representative examples to the 'examples' list\n"
            "6. Save as error_taxonomy.json (overwrite this template)"
        ),
        "statistics": {
            "total_train_samples": len(df),
            "total_errors": len(all_errors),
            "error_rate": float(len(all_errors) / len(df)),
            "false_negatives_total": len(false_negatives),
            "false_positives_total": len(false_positives),
            "false_negatives_high_conf": len(fn_high),
            "false_positives_high_conf": len(fp_high),
            "false_negatives_medium_conf": len(fn_medium),
            "false_positives_medium_conf": len(fp_medium),
            "high_conf_error_rate": float((len(fn_high) + len(fp_high)) / len(df)),
        },
        "generated_at": timestamp,
    }

    taxonomy_path = output_dir / "error_taxonomy_template.json"
    with open(taxonomy_path, "w") as f:
        json.dump(taxonomy, f, indent=2)

    logger.info(f"Saved error taxonomy template: {taxonomy_path}")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Error mining complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Manually inspect false_negatives_high_conf.csv")
    logger.info("  2. Categorize ~50 errors using error_taxonomy_template.json")
    logger.info("  3. Use categorized errors to guide CSE / DDA augmentation")
    logger.info("  4. Proceed to Qwen2.5-14B training with augmented data")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
