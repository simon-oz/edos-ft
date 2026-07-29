#!/usr/bin/env python3
# src/ensemble_rf_lr_transformer.py
"""
Ensemble: TF-IDF + LogisticRegression, TF-IDF + RandomForest, Transformer embeddings + LogisticRegression
Stacked using sklearn. StackingClassifier (stack_method='predict_proba').

Reads:
  data/processed/task_a_train.csv
  data/processed/task_a_dev.csv
  data/processed/task_a_test.csv

Writes:
  models/ensemble/stacking_pipeline.joblib
  models/ensemble/{tfidf_lr,tfidf_rf,embed_lr}_pipeline.joblib
  models/ensemble/predictions_test_{timestamp}.csv
  logs/ensemble_train_{timestamp}.log

Notes:
- Uses sentence-transformers to produce dense embeddings for the transformer branch.
- All base estimators expose predict_proba so StackingClassifier can use predict_proba as stacking features.
- Uses class_weight='balanced' where supported to mitigate class imbalance.
- Adjust hyperparameters (n_estimators, max_features, model_name) as needed.
"""

from pathlib import Path
from datetime import datetime
import argparse
import logging
import pandas as pd
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError(
        "sentence-transformers is required for transformer embeddings. "
        "Install with: pip install sentence-transformers"
    ) from e

# -------------------------
# Utilities
# -------------------------
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


# -------------------------
# sklearn-compatible transformer for sentence-transformers
# -------------------------
from sklearn.base import BaseEstimator, TransformerMixin

class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):
    """
    Wraps sentence-transformers SentenceTransformer to provide a .fit/.transform
    interface compatible with sklearn pipelines.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64, device: str = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device  # e.g., "cuda" or "cpu"
        self.model = None

    def fit(self, X, y=None):
        # lazy load model
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def transform(self, X):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        # X is iterable of strings
        embeddings = self.model.encode(
            X,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        return embeddings


# -------------------------
# Fit functions
# -------------------------
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
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state
    )
    return Pipeline([("tfidf", tfidf), ("rf", rf)])


def build_embed_lr(model_name="all-MiniLM-L6-v2", C=1.0, max_iter=20000, device=None):
    embed = SentenceTransformerVectorizer(model_name=model_name, device=device)
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, solver="lbfgs")
    return Pipeline([("embed", embed), ("lr", lr)])


# -------------------------
# Main orchestration
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
    parser.add_argument("--device", type=str, default=None, help="device for sentence-transformers (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--save_individual", action="store_true", help="save individual base pipelines")
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

    # load data
    logger.info("Loading datasets...")
    train_df = load_csv(Path(args.train))
    dev_df = load_csv(Path(args.dev))
    test_df = load_csv(Path(args.test))
    logger.info(f"Train samples: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    n_sexist = int((train_df["label"] == 1).sum())
    n_not = int((train_df["label"] == 0).sum())
    logger.info(f"Train class distribution: sexist={n_sexist}, not_sexist={n_not}")

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

    # Optionally save base pipelines before training (they are unfitted)
    if args.save_individual:
        joblib.dump(tfidf_lr, out_dir / "tfidf_lr_unfitted.joblib")
        joblib.dump(tfidf_rf, out_dir / "tfidf_rf_unfitted.joblib")
        joblib.dump(embed_lr, out_dir / "embed_lr_unfitted.joblib")
        logger.info("Saved unfitted base pipelines.")

    # StackingClassifier requires base estimators that support predict_proba when stack_method='predict_proba'
    # Our pipelines end with classifiers that implement predict_proba (LogisticRegression, RandomForest).
    estimators = [
        ("tfidf_lr", tfidf_lr),
        ("tfidf_rf", tfidf_rf),
        ("embed_lr", embed_lr),
    ]

    logger.info("Constructing StackingClassifier (stack_method='predict_proba')...")
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=20000),
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )

    # Fit stacking ensemble
    logger.info("Fitting stacking ensemble (this will fit all base estimators and the final estimator)...")
    X_train = train_df["text"].tolist()
    y_train = train_df["label"].values
    stacking.fit(X_train, y_train)
    logger.info("Stacking ensemble training complete.")

    # Evaluate on dev
    logger.info("Evaluating on dev set...")
    dev_pred = stacking.predict(dev_df["text"].tolist())
    evaluate(dev_df["label"].values, dev_pred, logger, prefix="Dev")

    # Evaluate on test
    logger.info("Evaluating on test set...")
    test_pred = stacking.predict(test_df["text"].tolist())
    evaluate(test_df["label"].values, test_pred, logger, prefix="Test")

    # Save models
    logger.info("Saving trained stacking pipeline and base estimators...")
    joblib.dump(stacking, out_dir / "stacking_pipeline.joblib")
    # Save fitted base estimators separately for inspection
    # Access fitted estimators via stacking.named_estimators_
    for name, _ in estimators:
        fitted = stacking.named_estimators_[name]
        joblib.dump(fitted, out_dir / f"{name}_fitted.joblib")
    logger.info(f"Saved artifacts to: {out_dir.resolve()}")

    # Save predictions CSV
    preds_df = test_df.copy()
    preds_df["pred"] = test_pred
    preds_csv = out_dir / f"predictions_test_{timestamp}.csv"
    preds_df.to_csv(preds_csv, index=False)
    logger.info(f"Saved test predictions to: {preds_csv}")

    logger.info("Ensemble training and evaluation complete.")


if __name__ == "__main__":
    main()
