#!/usr/bin/env python3
"""
ensemble_vote.py  (heterogeneous probability voter)
Globs per-member P(sexist) vectors written by train_encoder_task_a.py and learns a
WEIGHTED average + decision threshold that maximises macro-F1 on the CLEAN DEV set.
The 4k TEST set is reported ONCE at the end as the final look (eval hygiene: never
iterate on it -- the dev number is your working metric).

Why weighted (not plain mean): a plain mean gives every member equal say, including a
weak one. Optimising weights lets a strong member dominate while a complementary weak
member still contributes on the examples it uniquely gets right -- this is where the
diversity gain over the k-fold (same-architecture) ensemble actually lives.

Robustness:
  * Uses scipy differential_evolution (global, gradient-free) because macro-F1 is
    non-smooth in (weights, threshold); falls back to a uniform-weight + threshold grid
    if scipy is unavailable, so it never hard-fails.
  * K=1 still works (degenerates to single-model threshold tuning).
  * Reports equal-weight mean and best-single-member alongside the optimised vote, so
    you can SEE whether weighting/diversity is actually buying anything.

Usage (after training >=1 members with train_encoder_task_a.py):
  python src/ensemble_vote.py                       # use all *_dev_probs.npy found
  python src/ensemble_vote.py --members deberta_dapt,twhin   # subset
"""
import sys, json, glob, logging, argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
PROC_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_PROBS_DIR = PROJECT_ROOT / "models" / "ensemble_probs"
OUT_DIR = PROJECT_ROOT / "models" / "ensemble_vote"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"ensemble_vote_{timestamp}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")

THRESH_GRID = np.arange(0.05, 0.95, 0.005)


def macro_f1_at(probs, labels, t):
    return f1_score(labels, (probs >= t).astype(int), average="macro", zero_division=0)


def best_threshold(probs, labels):
    """Return (threshold, f1_macro) by grid search."""
    f = np.array([macro_f1_at(probs, labels, t) for t in THRESH_GRID])
    i = int(np.argmax(f)); return float(THRESH_GRID[i]), float(f[i])


def metrics(probs, labels, t):
    preds = (probs >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    pc = f1_score(labels, preds, average=None, zero_division=0)
    return {"f1_macro": float(f1), "accuracy": float((preds == labels).mean()),
            "f1_sexist": float(pc[1]) if len(pc) > 1 else 0.0,
            "f1_not_sexist": float(pc[0]) if len(pc) > 0 else 0.0,
            "n_pred_sexist": int(preds.sum()), "threshold": float(t)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--probs_dir", type=str, default=str(DEFAULT_PROBS_DIR))
    p.add_argument("--members", type=str, default=None,
                   help="Comma-separated subset of tags (default: all found)")
    p.add_argument("--dev_csv", type=str, default=str(PROC_DIR / "task_a_dev.csv"))
    p.add_argument("--test_csv", type=str, default=str(PROC_DIR / "task_a_test.csv"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    probs_dir = Path(args.probs_dir)

    # ---- discover members (tag -> (dev_probs, test_probs)) ----
    dev_files = sorted(glob.glob(str(probs_dir / "*_dev_probs.npy")))
    members = {}
    for df in dev_files:
        tag = Path(df).name[: -len("_dev_probs.npy")]
        tf = probs_dir / f"{tag}_test_probs.npy"
        if not tf.exists():
            logger.warning(f"Skip {tag}: no matching test probs at {tf}"); continue
        members[tag] = (np.load(df), np.load(tf))
    if args.members:
        want = [m.strip() for m in args.members.split(",")]
        missing = [m for m in want if m not in members]
        if missing:
            logger.error(f"Requested members not found in {probs_dir}: {missing} "
                         f"(available: {sorted(members)})"); sys.exit(1)
        members = {m: members[m] for m in want}
    if not members:
        logger.error(f"No probability vectors found in {probs_dir}. "
                     f"Run train_encoder_task_a.py first."); sys.exit(1)
    tags = list(members.keys()); K = len(tags)
    logger.info("=" * 60)
    logger.info(f"Members ({K}): {tags}")
    logger.info("=" * 60)

    # ---- labels from CSVs (deterministic; never trust saved label npy) ----
    dev_labels = pd.read_csv(args.dev_csv)["label"].astype(int).to_numpy()
    test_labels = pd.read_csv(args.test_csv)["label"].astype(int).to_numpy()
    dev_mat = np.stack([members[t][0] for t in tags], axis=0)    # (K, Ndev)
    test_mat = np.stack([members[t][1] for t in tags], axis=0)   # (K, Ntest)
    for t in tags:
        assert len(members[t][0]) == len(dev_labels), f"{t} dev probs length != dev labels"
        assert len(members[t][1]) == len(test_labels), f"{t} test probs length != test labels"

    def ens(mat, w):
        w = np.asarray(w, dtype=np.float64); w = w / w.sum()
        return w @ mat  # (N,)

    # ---- baselines: best single member, equal-weight mean ----
    logger.info("--- per-member (dev, threshold-tuned) ---")
    single_best_tag, single_best_f1 = None, -1.0
    for t in tags:
        th, f = best_threshold(members[t][0], dev_labels)
        logger.info(f"  {t:16s}: dev f1_macro={f:.4f} @ t={th:.3f}")
        if f > single_best_f1: single_best_f1, single_best_tag = f, t
    eq_w = np.ones(K) / K
    eq_th, eq_f = best_threshold(ens(dev_mat, eq_w), dev_labels)
    logger.info(f"  {'EQUAL-WEIGHT':16s}: dev f1_macro={eq_f:.4f} @ t={eq_th:.3f}")
    logger.info(f"  best single member = {single_best_tag} ({single_best_f1:.4f})")

    # ---- optimise weights + threshold on CLEAN DEV (global, gradient-free) ----
    def objective(x):
        logits, t = x[:-1], x[-1]
        w = np.exp(logits - logits.max()); w = w / w.sum()      # softmax -> simplex
        return -macro_f1_at(ens(dev_mat, w), dev_labels, t)     # minimise -F1

    opt_w, opt_t, opt_f = eq_w.copy(), eq_th, eq_f
    try:
        from scipy.optimize import differential_evolution
        bounds = [(-4.0, 4.0)] * K + [(0.05, 0.95)]
        res = differential_evolution(objective, bounds, seed=42, popsize=24,
                                     maxiter=80, tol=1e-4, polish=True,
                                     init="sobol", updating="immediate")
        logits, t = res.x[:-1], res.x[-1]
        w = np.exp(logits - logits.max()); w = w / w.sum()
        f = -res.fun
        if f > opt_f: opt_w, opt_t, opt_f = w, t, f
        logger.info(f"differential_evolution converged: fun={res.fun:.4f}")
    except Exception as e:
        logger.warning(f"scipy optimisation unavailable ({e!r}); using equal weights.")

    # fine threshold polish around optimum
    opt_th, opt_f2 = best_threshold(ens(dev_mat, opt_w), dev_labels)
    if opt_f2 > opt_f: opt_t, opt_f = opt_th, opt_f2

    logger.info("=" * 60); logger.info("OPTIMISED VOTE (weights + threshold tuned on CLEAN DEV)")
    logger.info("=" * 60)
    for t, wv in zip(tags, opt_w):
        logger.info(f"  weight {t:16s}: {wv:.4f}")
    logger.info(f"  threshold        : {opt_t:.3f}")
    dev_m = metrics(ens(dev_mat, opt_w), dev_labels, opt_t)
    logger.info("  DEV (working metric):")
    for k, v in dev_m.items(): logger.info(f"    {k:14s}: {v:.4f}" if isinstance(v, float) else f"    {k:14s}: {v}")

    # ---- TEST: single final look ----
    logger.info("=" * 60)
    logger.info("TEST -- FINAL LOOK (do NOT iterate on this number)")
    logger.info("=" * 60)
    test_ens = ens(test_mat, opt_w)
    test_m = metrics(test_ens, test_labels, opt_t)
    for k, v in test_m.items(): logger.info(f"  {k:14s}: {v:.4f}" if isinstance(v, float) else f"  {k:14s}: {v}")
    logger.info("\n" + classification_report(
        test_labels, (test_ens >= opt_t).astype(int),
        target_names=["not_sexist", "sexist"], digits=4))

    # ---- save: weights, summary, and a test preds CSV for error analysis ----
    np.save(OUT_DIR / "vote_test_probs.npy", test_ens)
    np.save(OUT_DIR / "vote_dev_probs.npy", ens(dev_mat, opt_w))
    test_df = pd.read_csv(args.test_csv)
    for t in tags: test_df[f"prob_{t}"] = members[t][1]
    test_df["prob_ensemble"] = test_ens
    test_df["pred_ensemble"] = (test_ens >= opt_t).astype(int)
    test_df.to_csv(OUT_DIR / f"vote_test_predictions_{timestamp}.csv", index=False)
    summary = {"members": tags, "weights": {t: float(wv) for t, wv in zip(tags, opt_w)},
               "threshold": float(opt_t),
               "equal_weight_dev_f1": float(eq_f), "best_single": single_best_tag,
               "best_single_dev_f1": float(single_best_f1),
               "optimised_dev_f1_macro": dev_m["f1_macro"],
               "optimised_dev_f1_sexist": dev_m["f1_sexist"],
               "test_f1_macro": test_m["f1_macro"], "test_f1_sexist": test_m["f1_sexist"],
               "test_f1_not_sexist": test_m["f1_not_sexist"], "test_accuracy": test_m["accuracy"],
               "timestamp": timestamp}
    json.dump(summary, open(OUT_DIR / "vote_summary.json", "w"), indent=2)
    logger.info(f"Saved weights/preds/summary to {OUT_DIR}")
    logger.info("Ensemble voting complete.")