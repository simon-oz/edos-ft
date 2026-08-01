#!/usr/bin/env python3
# src/train_logreg_task_b.py
"""
Train a TF-IDF + Logistic Regression classifier for EDOS Task B (4-category classification).

Uses:
- TfidfVectorizer for n-gram features.
- LogisticRegression with class_weight='balanced' to handle class imbalance.
"""

from pathlib import Path
from datetime import datetime
import argparse
import logging
import sys
import time

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import joblib


def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ],
    )
    return logging.getLogger(__name__)


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain 'text' and 'label' columns.")
    df["label"] = df["label"].astype(int)
    return df


def evaluate_and_log(y_true, y_pred, logger, target_names, dataset_name="Test"):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    logger.info(f"--- {dataset_name} basic metrics ---")
    logger.info(f"Accuracy         : {acc:.4f}")
    logger.info(f"Precision (macro): {prec:.4f}")
    logger.info(f"Recall    (macro): {rec:.4f}")
    logger.info(f"F1        (macro): {f1:.4f}")
    
    # Single, clear classification report
    logger.info(f"{dataset_name} Classification report (computed from the {dataset_name.lower()} dataset):")
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
    logger.info("\n" + report)
    
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"{dataset_name} Confusion matrix:\n{cm}")
    
    # Multi-class prediction distribution
    pred_dist = {target_names[i]: int((y_pred == i).sum()) for i in range(len(target_names))}
    logger.info(f"{dataset_name} prediction distribution: {pred_dist}")
    
    return {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1, "confusion_matrix": cm}


def fit_logreg(
    texts,
    labels,
    *,
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english",
    max_iter=1000,
    class_weight="balanced",
    C=1.0,
):
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        strip_accents="unicode",
        lowercase=True,
        stop_words=stop_words,
    )
    # Logistic Regression with L2 regularization, balanced class weights
    clf = LogisticRegression(
        class_weight=class_weight, 
        max_iter=max_iter,
        C=C,
        n_jobs=-1,  # Use all available CPU cores
        random_state=42
    )
    pipeline = Pipeline([("tfidf", tfidf), ("clf", clf)])
    pipeline.fit(texts, labels)
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression for EDOS Task B")
    parser.add_argument("--train", default="data/processed/task_b_train.csv")
    parser.add_argument("--dev", default="data/processed/task_b_dev.csv")
    parser.add_argument("--test", default="data/processed/task_b_test.csv")
    parser.add_argument("--out_dir", default="models/logreg_task_b")
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--stop_words", default="english", help="stop words for TfidfVectorizer or 'none'")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for Logistic Regression convergence")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # logging
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"logreg_task_b_train_{timestamp}.log"
    logger = setup_logger(log_file)
    logger.info(f"Logging to: {log_file}")
    logger.info(f"Arguments: {args}")

    # load data
    logger.info("Loading datasets...")
    train_df = load_csv(Path(args.train))
    dev_df = load_csv(Path(args.dev))
    test_df = load_csv(Path(args.test))
    logger.info(f"Train samples: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    # Extract target names dynamically from the training set
    if "label_category" not in train_df.columns:
        raise ValueError("Training CSV must contain 'label_category' column to map integer labels to names.")
    
    # Create mapping {0: "1. threats...", 1: "2. derogation", ...}
    label_map = train_df.drop_duplicates("label").set_index("label")["label_category"].to_dict()
    target_names = [label_map[i] for i in sorted(label_map.keys())]
    logger.info(f"Detected categories: {target_names}")

    train_dist = {target_names[i]: int((train_df["label"] == i).sum()) for i in range(len(target_names))}
    logger.info(f"Train class distribution: {train_dist}")

    stop_words = None if str(args.stop_words).lower() == "none" else args.stop_words

    # TRAIN with timing
    logger.info("Training classifier via fit_logreg(...)...")
    t0 = time.time()
    pipeline = fit_logreg(
        train_df["text"].tolist(),
        train_df["label"].values,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        stop_words=stop_words,
        max_iter=args.max_iter,
        class_weight="balanced",
        C=args.C,
    )
    t1 = time.time()
    train_time = t1 - t0
    logger.info(f"Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes).")

    # evaluate on dev with timing
    logger.info("Evaluating on dev set...")
    t0 = time.time()
    dev_pred = pipeline.predict(dev_df["text"].tolist())
    t1 = time.time()
    dev_time = t1 - t0
    logger.info(f"Dev prediction completed in {dev_time:.2f} seconds ({dev_time/60:.2f} minutes).")
    evaluate_and_log(dev_df["label"].values, dev_pred, logger, target_names, dataset_name="Dev")

    # evaluate on test with timing
    logger.info("Evaluating on test set...")
    t0 = time.time()
    test_pred = pipeline.predict(test_df["text"].tolist())
    t1 = time.time()
    test_time = t1 - t0
    logger.info(f"Test prediction completed in {test_time:.2f} seconds ({test_time/60:.2f} minutes).")
    evaluate_and_log(test_df["label"].values, test_pred, logger, target_names, dataset_name="Test")

    # save pipeline and predictions
    model_path = out_dir / "logreg_task_b_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved pipeline to: {model_path}")

    preds_df = test_df.copy()
    preds_df["pred"] = test_pred
    preds_csv = out_dir / f"predictions_test_{timestamp}.csv"
    preds_df.to_csv(preds_csv, index=False)
    logger.info(f"Saved test predictions to: {preds_csv}")

    # summary timings
    logger.info("=== Timing summary ===")
    logger.info(f"Training time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    logger.info(f"Dev prediction time: {dev_time:.2f} seconds ({dev_time/60:.2f} minutes)")
    logger.info(f"Test prediction time: {test_time:.2f} seconds ({test_time/60:.2f} minutes)")

    logger.info("Training complete.")
    logger.info(f"Saved artifacts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()