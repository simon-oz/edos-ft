#!/usr/bin/env python3
"""probe_lexical_ceiling_task_c.py — measure the TF-IDF ceiling on EDOS Task C.

Task-C version of probe_lexical_ceiling_task_b.py: same word+char TF-IDF features and
the same three lexical learners (LR, LinearSVC, XGBoost); only the default CSV paths
change. If all three agree around the same macro-F1, that agreement IS the measured
lexical ceiling for Task C — the gap to your transformers is representation, not
learner choice.

Usage:
  python src/probe_lexical_ceiling_task_c.py
"""
import time, argparse
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import xgboost as xgb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/task_c_train.csv")
    ap.add_argument("--dev", default="data/processed/task_c_dev.csv")
    ap.add_argument("--test", default="data/processed/task_c_test.csv")
    a = ap.parse_args()
    tr, dv, te = (pd.read_csv(p) for p in (a.train, a.dev, a.test))
    ytr, ydv, yte = tr["label"].values, dv["label"].values, te["label"].values
    print(f"Train={len(tr)} Dev={len(dv)} Test={len(te)} num_classes={int(ytr.max()) + 1}")
    print(f"Train class distribution: {np.bincount(ytr).tolist()}")

    word = TfidfVectorizer(max_features=100000, ngram_range=(1, 2), sublinear_tf=True,
                           strip_accents="unicode")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=200000,
                           sublinear_tf=True)
    Xtr = hstack([word.fit_transform(tr["text"]), char.fit_transform(tr["text"])])
    Xdv = hstack([word.transform(dv["text"]), char.transform(dv["text"])])
    Xte = hstack([word.transform(te["text"]), char.transform(te["text"])])

    counts = np.bincount(ytr)
    sw = len(ytr) / (len(counts) * counts[ytr])   # balanced sample weights

    models = {
        "lr": LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"),
        "linearsvc": LinearSVC(C=0.5, max_iter=5000, class_weight="balanced"),
        "xgb": xgb.XGBClassifier(n_estimators=800, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.3, min_child_weight=5,
                                  reg_lambda=5.0, eval_metric="mlogloss",
                                  early_stopping_rounds=50, n_jobs=8),
    }
    for name, m in models.items():
        t0 = time.time()
        if name == "xgb":
            m.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xdv, ydv)], verbose=False)
        else:
            m.fit(Xtr, ytr)
        fd = f1_score(ydv, m.predict(Xdv), average="macro", zero_division=0)
        ft = f1_score(yte, m.predict(Xte), average="macro", zero_division=0)
        print(f"{name:10s} dev f1_macro={fd:.4f} test f1_macro={ft:.4f} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()