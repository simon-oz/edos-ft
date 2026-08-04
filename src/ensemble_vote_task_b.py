#!/usr/bin/env python3
"""ensemble_vote_task_b.py — weighted multi-class vote over Task-B (N,4) prob members."""
import argparse, json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, classification_report
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC = PROJECT_ROOT / "data" / "processed"
PROBS = PROJECT_ROOT / "models" / "ensemble_probs"
OUT = PROJECT_ROOT / "models" / "ensemble_vote_task_b"; OUT.mkdir(parents=True, exist_ok=True)

def macro_f1(probs, labels):
    return f1_score(labels, probs.argmax(1), average="macro", zero_division=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default=None,
                    help="comma-separated tags; default = all *_task_b_dev_probs.npy found")
    a = ap.parse_args()
    members = {}
    for f in sorted(glob.glob(str(PROBS / "*_task_b_dev_probs.npy"))):
        tag = Path(f).name.replace("_task_b_dev_probs.npy", "")
        tf = PROBS / f"{tag}_task_b_test_probs.npy"
        if tf.exists():
            members[tag] = (np.load(f), np.load(tf))
    if a.members:
        want = [m.strip() for m in a.members.split(",")]
        miss = [m for m in want if m not in members]
        if miss: print(f"missing: {miss}; available: {sorted(members)}"); sys.exit(1)
        members = {m: members[m] for m in want}
    tags = list(members); K = len(tags)
    dev_labels = pd.read_csv(PROC / "task_b_dev.csv")["label"].to_numpy()
    test_labels = pd.read_csv(PROC / "task_b_test.csv")["label"].to_numpy()
    devM = np.stack([members[t][0] for t in tags])    # (K, N, 4)
    testM = np.stack([members[t][1] for t in tags])

    print("--- per-member (dev, argmax) ---")
    for t in tags: print(f"  {t:16s}: dev f1_macro={macro_f1(members[t][0], dev_labels):.4f}")
    eq = devM.mean(0)
    print(f"  {'EQUAL-WEIGHT':16s}: dev f1_macro={macro_f1(eq, dev_labels):.4f}")

    def neg_f1(x):
        w = np.exp(x - x.max()); w = w / w.sum()
        return -macro_f1((devM * w[:, None, None]).sum(0), dev_labels)
    res = differential_evolution(neg_f1, [(-4, 4)] * K, seed=42, popsize=20, maxiter=80, tol=1e-5)
    w = np.exp(res.x - res.x.max()); w = w / w.sum()

    print("--- OPTIMISED VOTE (weights tuned on CLEAN DEV) ---")
    for t, wi in zip(tags, w): print(f"  weight {t:16s}: {wi:.4f}")
    dev_opt = (devM * w[:, None, None]).sum(0); test_opt = (testM * w[:, None, None]).sum(0)
    print(f"  DEV  f1_macro (working metric): {macro_f1(dev_opt, dev_labels):.4f}")
    print(f"  TEST f1_macro (FINAL LOOK)    : {macro_f1(test_opt, test_labels):.4f}")
    print(classification_report(test_labels, test_opt.argmax(1), digits=4, zero_division=0))
    json.dump({"members": tags, "weights": {t: float(x) for t, x in zip(tags, w)},
               "dev_f1_macro": float(macro_f1(dev_opt, dev_labels)),
               "test_f1_macro": float(macro_f1(test_opt, test_labels))},
              open(OUT / "vote_summary.json", "w"), indent=2)

if __name__ == "__main__":
    main()