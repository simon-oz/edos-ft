#!/usr/bin/env python3
# src/ensemble_rf_lr_transformer.py
"""
Ensemble: TF-IDF + LogisticRegression, TF-IDF + RandomForest, Transformer embeddings + LogisticRegression
Stacked using sklearn StackingClassifier.

Changes:
- Adds timing for fitting the stacking ensemble and for dev/test prediction.
- Prints a single, clearly labeled classification report and confusion matrix per evaluation dataset.
- Saves fitted base estimators and stacking pipeline.
"""

from pathlib import Path
from datetime import datetime
import argparse
import logging
import pandas as pd
import joblib
import numpy as np
import time
import sys

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers") from e

from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------
# Utilities
# -------------------------
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


class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64, device: str = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None

    def fit(self, X, y=None):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def transform(self, X):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        embeddings = self.model.encode(
            X,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        return embeddings


def build_tfidf_lr(max_features=50000, ngram_range=(1,2), stop_words="english", C=1.0, max_iter=20000):
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        strip_accents="unicode",
        lowercase=True,
        stop_words=stop_words
    )
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, solver="lbfgs")
    return Pipeline([("tfidf", tfidf), ("lr", lr)])


def build_tfidf_rf(max_features=50000, ngram_range=(1,2), stop_words="english", n_estimators=300, max_features_rf="sqrt", random_state=42):
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        strip_accents="unicode",
        lowercase=True,
        stop_words=stop_words
    )
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features_rf,
        class_weight='balanced',
        n_jobs=-1,
        random_state=random_state
    )
    return Pipeline([("tfidf", tfidf), ("rf", rf)])


def build_embed_lr(model_name="all-MiniLM-L6-v2", C=1.0, max_iter=20000, device=None):
    embed = SentenceTransformerVectorizer(model_name=model_name, device=device)
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, solver="lbfgs")
    return Pipeline([("embed", embed), ("lr", lr)])


def evaluate_and_log_clf(estimator, X, y, logger, dataset_name="Test"):
    t0 = time.time()
    preds = estimator.predict(X)
    t1 = time.time()
    elapsed = t1 - t0
    acc = accuracy_score(y, preds)
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    logger.info(f"{dataset_name} basic metrics: accuracy={acc:.4f}, f1_macro={f1m:.4f}")
    logger.info(f"{dataset_name} Classification report (computed from the {dataset_name.lower()} dataset):")
    report = classification_report(y, preds, target_names=["not_sexist", "sexist"], digits=4, zero_division=0)
    logger.info("\n" + report)
    logger.info(f"{dataset_name} Confusion matrix:\n{confusion_matrix(y, preds)}")
    logger.info(f"{dataset_name} prediction time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    return {"preds": preds, "time": elapsed}


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Ensemble (RF + LR + Transformer) for EDOS Task A")
    parser.add_argument("--train", default="data/processed/task_a_train.csv")
    parser.add_argument("--dev", default="data/processed/task_a_dev.csv")
    parser.add_argument("--test", default="data/processed/task_a_test.csv")
    parser.add_argument("--out_dir", default="models/ensemble")
    parser.add_argument("--tfidf_max_features", type=int, default=50000)
    parser.add_argument("--tfidf_ngram_min", type=int, default=1)
    parser.add_argument("--tfidf_ngram_max", type=int, default=2)
    parser.add_argument("--rf_n_estimators", type=int, default=300)
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_individual", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # logging
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"ensemble_train_{timestamp}.log"
    logger = setup_logger(log_file)
    logger.info(f"Logging to: {log_file}")
    logger.info(f"Arguments: {args}")

    # load data
    logger.info("Loading datasets...")
    train = pd.read_csv(args.train, dtype=str, keep_default_na=False)
    dev = pd.read_csv(args.dev, dtype=str, keep_default_na=False)
    test = pd.read_csv(args.test, dtype=str, keep_default_na=False)
    for df in (train, dev, test):
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV files must contain 'text' and 'label' columns.")
        df["label"] = df["label"].astype(int)
    logger.info(f"Train samples: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")
    logger.info(f"Train class distribution: {train['label'].value_counts().to_dict()}")

    # Build base pipelines
    logger.info("Building base pipelines...")
    tfidf_lr = build_tfidf_lr(
        max_features=args.tfidf_max_features,
        ngram_range=(args.tfidf_ngram_min, args.tfidf_ngram_max),
    )
    tfidf_rf = build_tfidf_rf(
        max_features=args.tfidf_max_features,
        ngram_range=(args.tfidf_ngram_min, args.tfidf_ngram_max),
        n_estimators=args.rf_n_estimators,
    )
    embed_lr = build_embed_lr(model_name=args.embed_model, device=args.device)

    estimators = [
        ("tfidf_lr", tfidf_lr),
        ("tfidf_rf", tfidf_rf),
        ("embed_lr", embed_lr),
    ]

    # Fit stacking ensemble with timing
    logger.info("Constructing StackingClassifier...")
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=20000),
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )

    X_train = train["text"].tolist()
    y_train = train["label"].values

    logger.info("Fitting stacking ensemble (this will fit all base estimators and the final estimator)...")
    t0 = time.time()
    stacking.fit(X_train, y_train)
    t1 = time.time()
    fit_time = t1 - t0
    logger.info(f"Stacking ensemble fitted in {fit_time:.2f} seconds ({fit_time/60:.2f} minutes).")

    # Optionally save fitted base estimators and stacking pipeline
    logger.info("Saving fitted stacking pipeline and base estimators...")
    joblib.dump(stacking, out_dir / "stacking_pipeline.joblib")
    for name, _ in estimators:
        fitted = stacking.named_estimators_[name]
        joblib.dump(fitted, out_dir / f"{name}_fitted.joblib")
    logger.info(f"Saved artifacts to: {out_dir.resolve()}")

    # Evaluate on dev
    logger.info("Evaluating on dev set...")
    dev_eval = evaluate_and_log_clf(stacking, dev["text"].tolist(), dev["label"].values, logger, dataset_name="Dev")

    # Evaluate on test
    logger.info("Evaluating on test set...")
    test_eval = evaluate_and_log_clf(stacking, test["text"].tolist(), test["label"].values, logger, dataset_name="Test")

    # Save test predictions CSV
    logger.info("Saving test predictions CSV...")
    test_preds = test_eval["preds"]
    # If you want probabilities, compute them via predict_proba
    try:
        probs = stacking.predict_proba(test["text"].tolist())
        prob_not = probs[:, 0]
        prob_sex = probs[:, 1]
    except Exception:
        prob_not = [None] * len(test_preds)
        prob_sex = [None] * len(test_preds)

    test_out = test.copy().reset_index(drop=True)
    test_out["pred"] = test_preds
    test_out["prob_not_sexist"] = prob_not
    test_out["prob_sexist"] = prob_sex
    preds_csv = out_dir / f"predictions_test_{timestamp}.csv"
    test_out.to_csv(preds_csv, index=False)
    logger.info(f"Saved test predictions to: {preds_csv}")

    # Timing summary
    logger.info("=== Timing summary ===")
    logger.info(f"Fitting time (stacking ensemble): {fit_time:.2f} seconds ({fit_time/60:.2f} minutes)")
    logger.info(f"Dev prediction time: {dev_eval['time']:.2f} seconds ({dev_eval['time']/60:.2f} minutes)")
    logger.info(f"Test prediction time: {test_eval['time']:.2f} seconds ({test_eval['time']/60:.2f} minutes)")

    logger.info("Ensemble training and evaluation complete.")


if __name__ == "__main__":
    main()
