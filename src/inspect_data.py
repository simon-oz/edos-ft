#!/usr/bin/env python3
"""
inspect_data.py
Quick sanity checks and statistics for all processed EDOS data.

Usage:
  From project root:  python src/inspect_data.py
  From src/ folder:   python inspect_data.py
"""

import pandas as pd
from pathlib import Path
import logging
import sys
from datetime import datetime

# ------------------------------------------------------------------
# Auto-detect project root
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

PROC_DIR = PROJECT_ROOT / "data" / "processed"

# ------------------------------------------------------------------
# Logging setup: console + file
# ------------------------------------------------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"inspect_data_{timestamp}.log"

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


def print_stats(name: str, df: pd.DataFrame, task: str = None):
    logger.info("-" * 60)
    logger.info(f"{name}: {len(df)} rows")
    if task == "A" and "label" in df.columns:
        c = df["label"].value_counts().sort_index()
        logger.info(f"  Labels: {dict(c)}")
        if len(c) == 2:
            logger.info(f"  Class balance (min/max): {c.min() / c.max():.3f}")
    elif task == "B" and "label_category" in df.columns:
        vc = df["label_category"].value_counts().sort_index()
        logger.info(f"  Categories ({len(vc)}):")
        for v, n in vc.items():
            logger.info(f"    {v}: {n}")
    elif task == "C" and "label_vector" in df.columns:
        vc = df["label_vector"].value_counts().sort_index()
        logger.info(f"  Vectors ({len(vc)} unique):")
        for v, n in vc.items():
            logger.info(f"    {v}: {n}")


def main():
    logger.info("=" * 60)
    logger.info("EDOS DATA INSPECTION")
    logger.info("=" * 60)

    # --- Aggregated ---
    agg_path = PROC_DIR / "edos_aggregated.csv"
    if agg_path.exists():
        df_agg = pd.read_csv(agg_path, dtype=str, keep_default_na=False)
        logger.info("[edos_aggregated.csv]")
        logger.info(f"  Total rows    : {len(df_agg)}")
        logger.info(f"  Splits        : {dict(df_agg['split'].value_counts().sort_index())}")
        logger.info(f"  Sexism rate   : {(df_agg['label_sexist'].str.strip() == 'sexist').mean():.3f}")
    else:
        logger.info(f"[SKIP] {agg_path} not found")

    # --- Individual annotations ---
    ind_path = PROC_DIR / "edos_individual.csv"
    if ind_path.exists():
        df_ind = pd.read_csv(ind_path, dtype=str, keep_default_na=False)
        logger.info("[edos_individual.csv]")
        logger.info(f"  Total annotations : {len(df_ind)}")
        logger.info(f"  Unique texts      : {df_ind['rewire_id'].nunique()}")
        logger.info(f"  Annotators        : {df_ind['annotator'].nunique()}")

        sample_id = df_ind["rewire_id"].iloc[0]
        sample = df_ind[df_ind["rewire_id"] == sample_id]
        logger.info(f"  Sample agreement for {sample_id}:")
        for _, row in sample.head(5).iterrows():
            logger.info(f"    Annotator {row['annotator']:>3s}: {row['label_sexist']:10s} / {row['label_vector']}")
    else:
        logger.info(f"[SKIP] {ind_path} not found")

    # --- Unlabeled ---
    for fname in ["unlabeled_gab.csv", "unlabeled_reddit.csv", "unlabeled_combined_400k.csv"]:
        fpath = PROC_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath, dtype=str, keep_default_na=False)
            logger.info(f"[{fname}]")
            logger.info(f"  Rows            : {len(df):,}")
            if "text" in df.columns:
                lengths = df["text"].astype(str).str.len()
                logger.info(f"  Avg text length : {lengths.mean():.1f} chars")
                logger.info(f"  Median length   : {lengths.median():.1f} chars")

    # --- Task datasets (Strategy A) ---
    task_dir = PROC_DIR / "task_datasets_a"
    if task_dir.exists():
        logger.info("[task_datasets_a/]")
        for task in ["A", "B", "C"]:
            for split in ["train", "custom_dev", "official_dev", "official_test"]:
                fpath = task_dir / f"task_{task.lower()}_{split}.csv"
                if fpath.exists():
                    df = pd.read_csv(fpath, dtype=str, keep_default_na=False)
                    print_stats(f"Task {task} / {split}", df, task)
    else:
        logger.info(f"[SKIP] {task_dir} not found — run build_task_datasets.py first")

    logger.info("=" * 60)
    logger.info("Inspection complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
