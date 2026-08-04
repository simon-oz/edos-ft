#!/usr/bin/env python3
"""ensemble_vote_task_c.py — weighted multi-class vote over Task-C (N, C) prob members.
Task-C generalization of ensemble_vote_task_b.py:
  * globs *_task_c_{dev,test}_probs.npy from models/ensemble_probs/
  * labels from task_c_{dev,test}.csv
  * num_classes read from the prob shape (10-11 vectors); target names from 'label_vector'
  * weights tuned on CLEAN DEV via differential_evolution; TEST is the single final look

Usage:
  python src/ensemble_vote_task_c.py                      # all *_task_c_*_probs.npy found
  python src/ensemble_vote_task_c.py --members deberta_task_c,qwen_task_c
"""
import argparse, json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, classification_report
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC = PROJECT_ROOT / "data" / "processed"
PROBS = PROJECT_ROOT / "models" / "ensemble_probs"
OUT = PROJECT_ROOT / "models" / "ensemble_vote_task_c"; OUT.mkdir(parents=True, exist_ok=True)

def macro_f1(probs, labels):
    return f1_score(labels, probs.argmax(1), average="macro", zero_division=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default=None,
                    help="comma-separated tags; default = all *_task_c_dev_probs.npy found")
    a = ap.parse_args()
    members = {}
    for f in sorted(glob.glob(str(PROBS / "*_task_c_dev_probs.npy"))):
        tag = Path(f).name.replace("_task_c_dev_probs.npy", "")
        tf = PROBS / f"{tag}_task_c_test_probs.npy"
        if tf.exists():
            members[tag] = (np.load(f), np.load(tf))
    if a.members:
        want = [m.strip() for m in a.members.split(",")]
        miss = [m for m in want if m not in members]
        if miss: print(f"missing: {miss}; available: {sorted(members)}"); sys.exit(1)
        members = {m: members[m] for m in want}
    if not members:
        print(f"No Task C prob members found in {PROBS}. "
              f"Run train_qwen_task_c.py and/or emit_deberta_task_c_probs.py first.")
        sys.exit(1)
    tags = list(members); K = len(tags)

    dev_df = pd.read_csv(PROC / "task_c_dev.csv")
    test_df = pd.read_csv(PROC / "task_c_test.csv")
    dev_labels = dev_df["label"].to_numpy()
    test_labels = test_df["label"].to_numpy()
    devM = np.stack([members[t][0] for t in tags])     # (K, N, C)
    testM = np.stack([members[t][1] for t in tags])
    num_classes = devM.shape[2]

    # sanity: consistent shapes across members and labels
    for t in tags:
        assert members[t][0].shape == (len(dev_labels), num_classes), f"{t} dev probs shape mismatch"
        assert members[t][1].shape == (len(test_labels), num_classes), f"{t} test probs shape mismatch"

    # target names from label_vector column (generic fallback)
    if "label_vector" in dev_df.columns:
        pairs = list(zip(dev_df["label"], dev_df["label_vector"]))
        label_map = {int(l): v for l, v in pairs}
        target_names = [str(label_map.get(i, f"class_{i}")) for i in range(num_classes)]
    else:
        target_names = [f"class_{i}" for i in range(num_classes)]

    print(f"--- Task C vote: {K} member(s), {num_classes} classes ---")
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
    dev_opt = (devM * w[:, None, None]).sum(0)
    test_opt = (testM * w[:, None, None]).sum(0)
    print(f"  DEV  f1_macro (working metric): {macro_f1(dev_opt, dev_labels):.4f}")
    print(f"  TEST f1_macro (FINAL LOOK)    : {macro_f1(test_opt, test_labels):.4f}")
    print(classification_report(test_labels, test_opt.argmax(1), target_names=target_names,
                                digits=4, zero_division=0))
    json.dump({"task": "Task C", "members": tags, "num_classes": num_classes,
               "weights": {t: float(x) for t, x in zip(tags, w)},
               "dev_f1_macro": float(macro_f1(dev_opt, dev_labels)),
               "test_f1_macro": float(macro_f1(test_opt, test_labels))},
              open(OUT / "vote_summary.json", "w"), indent=2)

if __name__ == "__main__":
    main()