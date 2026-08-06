#!/usr/bin/env python3
"""evaluate_llm_task_a.py — evaluate LLM Task-A predictions against EDOS gold labels.

The prediction file must contain an `id` column (matching `rewire_id` in the gold file)
and a `predicted` column with values like 'sexist' / 'none'.

Usage:
  python src/evaluate_llm_task_a.py --pred data/processed/llm_fewshot/taska/gemini3-1-pro/predicted-by-code.csv
  python src/evaluate_llm_task_a.py --pred data/processed/llm_fewshot/taska/chatgpt/test_id_text_1_predicted.csv
  python src/evaluate_llm_task_a.py --pred data/processed/llm_fewshot/taska/glm52/predicted.csv
  python src/evaluate_llm_task_a.py --pred <path> --split test --pred_col predicted-
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
TARGET_NAMES = ["not_sexist", "sexist"]


# ------------------------------------------------------------------
# Logging setup: console + file (mirrors train_svm_task_c.py)
# ------------------------------------------------------------------
def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def normalize_prediction(pred):
    """Map a raw LLM prediction to 1 (sexist), 0 (not sexist), or None (unparseable)."""
    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
        return None
    s = str(pred).strip().lower().strip('"\'` ')
    if s in ("sexist", "1", "true", "yes"):
        return 1
    if s in ("none", "not sexist", "not_sexist", "notsexist", "not-sexist", "0", "false", "no"):
        return 0
    # Fallback heuristics for verbose outputs
    if s.startswith("not") or s == "none":
        return 0
    if "sexist" in s:
        return 1
    return None


def normalize_gold(label):
    """Map gold label_sexist ('sexist' / 'not sexist') to 1/0."""
    return 1 if str(label).strip().lower() == "sexist" else 0


def main():
    ap = argparse.ArgumentParser(description="Evaluate LLM Task-A predictions on EDOS")
    ap.add_argument("--pred", required=True, help="Path to LLM prediction CSV (columns: id, predicted)")
    ap.add_argument("--gold", default=str(PROJECT_ROOT / "data" / "raw" / "edos_labelled_aggregated.csv"),
                    help="Path to gold labels CSV")
    ap.add_argument("--pred_col", default="predicted", help="Name of the prediction column")
    ap.add_argument("--split", default="test", help="Gold split to evaluate against (default: test)")
    args = ap.parse_args()

    # ---- logger (timestamped file under logs/) ----
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"evaluate_llm_task_a_{timestamp}.log"
    logger = setup_logger(log_file)
    logger.info(f"Logging to: {log_file}")
    logger.info(f"Arguments: {args}")

    # ---- load predictions ----
    pred_df = pd.read_csv(args.pred, dtype=str, keep_default_na=False)
    if "id" not in pred_df.columns or args.pred_col not in pred_df.columns:
        logger.error(f"Prediction file must have 'id' and '{args.pred_col}' columns; "
                     f"got {pred_df.columns.tolist()}")
        sys.exit(1)

    # ---- load gold, filter to split ----
    gold_df = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    gold_df = gold_df[gold_df["split"].str.strip().str.lower() == args.split].copy()
    gold_df["label_bin"] = gold_df["label_sexist"].apply(normalize_gold)

    logger.info(f"Predictions   : {len(pred_df)} rows  <- {args.pred}")
    logger.info(f"Gold ({args.split}) : {len(gold_df)} rows  <- {args.gold}")

    # ---- join on id == rewire_id ----
    merged = pred_df.merge(gold_df[["rewire_id", "label_bin"]],
                           left_on="id", right_on="rewire_id", how="inner")
    n_missing = len(gold_df) - len(merged)
    logger.info(f"Matched       : {len(merged)}  |  missing predictions: {n_missing}")

    # ---- normalize predictions ----
    merged["pred_bin"] = merged[args.pred_col].apply(normalize_prediction)
    n_invalid = int(merged["pred_bin"].isna().sum())
    valid = merged.dropna(subset=["pred_bin"]).copy()
    valid["pred_bin"] = valid["pred_bin"].astype(int)
    logger.info(f"Valid         : {len(valid)}  |  unparseable predictions: {n_invalid}")

    if len(valid) == 0:
        logger.error("No valid predictions to evaluate.")
        sys.exit(1)

    y_true = valid["label_bin"].values
    y_pred = valid["pred_bin"].values

    # ---- metrics ----
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    logger.info("=" * 60)
    logger.info(f"Task A evaluation on '{args.split}' split  ({len(valid)} valid predictions)")
    logger.info("=" * 60)
    logger.info(f"Accuracy          : {acc:.4f}")
    logger.info(f"F1 (macro)        : {f1_macro:.4f}")
    logger.info(f"Precision (macro) : {prec:.4f}")
    logger.info(f"Recall (macro)    : {rec:.4f}")
    logger.info("Classification report:")
    logger.info("\n" + classification_report(y_true, y_pred, target_names=TARGET_NAMES,
                                             digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    logger.info("Confusion matrix (rows=true, cols=predicted):")
    logger.info(f"                 pred_not_sexist  pred_sexist")
    logger.info(f"  true_not_sexist       {cm[0][0]:>6}       {cm[0][1]:>6}")
    logger.info(f"  true_sexist           {cm[1][0]:>6}       {cm[1][1]:>6}")

    if n_missing > 0 or n_invalid > 0:
        logger.info(f"NOTE: metrics above are on {len(valid)} valid predictions only. "
                    f"{n_missing} gold rows had no prediction and {n_invalid} predictions were unparseable.")


if __name__ == "__main__":
    main()