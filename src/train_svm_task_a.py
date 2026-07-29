#!/usr/bin/env python3
# src/train_svm_task_a.py
"""
Train a TF-IDF + LinearSVC classifier for EDOS Task A (binary sexism detection).

Refactored so the SVM training is encapsulated in a clear `fit_svm` function.

Reads:
  data/processed/task_a_train.csv
  data/processed/task_a_dev.csv
  data/processed/task_a_test.csv

Writes:
  models/svm/svm_pipeline.joblib
  models/svm/predictions_test_{timestamp}.csv
  logs/srv_svm_train_{timestamp}.log

Usage:
python src/train_svm_task_a.py

train_svm_task_a.py notable args
--train path to train CSV (default data/processed/task_a_train.csv)
--dev path to dev CSV (default data/processed/task_a_dev.csv)
--test path to test CSV (default data/processed/task_a_test.csv)
--out_dir output dir (default models/svm)
--max_features TF‑IDF max features (default 50000)
--ngram_min / --ngram_max ngram range (defaults 1 2)
--stop_words stop words or none (default english)
"""

from pathlib import Path
from datetime import datetime
import argparse
import logging
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)


def setup_logger(log_path: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler()
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


def evaluate(y_true, y_pred, logger, prefix="Test"):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=["not_sexist", "sexist"], digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"--- {prefix} results ---")
    logger.info(f"Accuracy         : {acc:.4f}")
    logger.info(f"Precision (macro): {prec:.4f}")
    logger.info(f"Recall    (macro): {rec:.4f}")
    logger.info(f"F1        (macro): {f1:.4f}")
    logger.info("Classification report:\n" + report)
    logger.info(f"Confusion matrix:\n{cm}")
    logger.info(f"Prediction distribution: not_sexist={int((y_pred==0).sum())}, sexist={int((y_pred==1).sum())}")
    return {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1, "confusion_matrix": cm}


def fit_svm(
    texts,
    labels,
    *,
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english",
    max_iter=20000,
    class_weight="balanced",
):
    """
    Fit and return a scikit-learn Pipeline: TF-IDF -> LinearSVC.

    Args:
      texts: iterable of raw text strings (training data).
      labels: iterable of integer labels (0/1).
      max_features: max features for TfidfVectorizer.
      ngram_range: tuple (min_n, max_n).
      stop_words: stop words setting for TfidfVectorizer or None.
      max_iter: max iterations for LinearSVC.
      class_weight: class_weight passed to LinearSVC (e.g., 'balanced' or dict).

    Returns:
      pipeline: fitted sklearn Pipeline object.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        strip_accents="unicode",
        lowercase=True,
        stop_words=stop_words,
    )
    clf = LinearSVC(class_weight=class_weight, max_iter=max_iter)
    pipeline = Pipeline([("tfidf", tfidf), ("clf", clf)])
    pipeline.fit(texts, labels)
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + LinearSVC for EDOS Task A (refactored)")
    parser.add_argument("--train", default="data/processed/task_a_train.csv")
    parser.add_argument("--dev", default="data/processed/task_a_dev.csv")
    parser.add_argument("--test", default="data/processed/task_a_test.csv")
    parser.add_argument("--out_dir", default="models/svm")
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--stop_words", default="english", help="stop words for TfidfVectorizer or 'none'")
    parser.add_argument("--max_iter", type=int, default=20000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # logging
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"srv_svm_train_{timestamp}.log"
    logger = setup_logger(log_file)
    logger.info(f"Logging to: {log_file}")

    # load data
    logger.info("Loading datasets...")
    train_df = load_csv(Path(args.train))
    dev_df = load_csv(Path(args.dev))
    test_df = load_csv(Path(args.test))
    logger.info(f"Train samples: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    n_sexist = int((train_df["label"] == 1).sum())
    n_not = int((train_df["label"] == 0).sum())
    logger.info(f"Train class distribution: sexist={n_sexist}, not_sexist={n_not}")

    # prepare stop_words argument
    stop_words = None if str(args.stop_words).lower() == "none" else args.stop_words

    # TRAIN: use the dedicated fit_svm function
    logger.info("Training classifier via fit_svm(...)...")
    pipeline = fit_svm(
        train_df["text"].tolist(),
        train_df["label"].values,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        stop_words=stop_words,
        max_iter=args.max_iter,
        class_weight="balanced",
    )

    # evaluate on dev
    logger.info("Evaluating on dev set...")
    dev_pred = pipeline.predict(dev_df["text"].tolist())
    evaluate(dev_df["label"].values, dev_pred, logger, prefix="Dev")

    # evaluate on test
    logger.info("Evaluating on test set...")
    test_pred = pipeline.predict(test_df["text"].tolist())
    evaluate(test_df["label"].values, test_pred, logger, prefix="Test")

    # save pipeline and predictions
    model_path = out_dir / "svm_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved pipeline to: {model_path}")

    preds_df = test_df.copy()
    preds_df["pred"] = test_pred
    preds_csv = out_dir / f"predictions_test_{timestamp}.csv"
    preds_df.to_csv(preds_csv, index=False)
    logger.info(f"Saved test predictions to: {preds_csv}")

    logger.info("Training complete.")
    logger.info(f"Saved artifacts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
