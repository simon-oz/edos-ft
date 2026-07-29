#!/usr/bin/env python3
"""
create_holdout_test.py
Since the SemEval competition is closed, this script creates a reliable
internal evaluation protocol by stratifying the official TRAIN split.

Strategy (RECOMMENDED):
  - Official TRAIN  → 80% new_train / 20% custom_dev (stratified)
  - Official DEV    → kept as validation / final-test alternative
  - Official TEST   → completely held out until FINAL evaluation

Output:
  data/processed/splits_strategy_a/
    ├── train.csv
    ├── custom_dev.csv
    ├── official_dev.csv
    └── official_test.csv

Usage:
  From project root:  python src/create_holdout_test.py
  From src/ folder:   python create_holdout_test.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import sys
import logging
from datetime import datetime

# ------------------------------------------------------------------
# Auto-detect project root
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

PROC_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROC_DIR / "splits_strategy_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Logging setup: console + file
# ------------------------------------------------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
script_name = Path(__file__).stem
log_file = LOG_DIR / f"{script_name}_{timestamp}.log"

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

# ------------------------------------------------------------------
# Load aggregated data
# ------------------------------------------------------------------
agg_path = PROC_DIR / "edos_aggregated.csv"
if not agg_path.exists():
    logger.error(f"{agg_path} not found. Run parse_edos_data.py first.")
    sys.exit(1)

df = pd.read_csv(agg_path, dtype=str, keep_default_na=False)
for col in ["label_sexist", "label_category", "label_vector", "split"]:
    if col in df.columns:
        df[col] = df[col].str.strip()

# ------------------------------------------------------------------
# Split official data by the 'split' column
# ------------------------------------------------------------------
train_df = df[df["split"] == "train"].copy()
dev_df   = df[df["split"] == "dev"].copy()
test_df  = df[df["split"] == "test"].copy()

logger.info(f"Original splits — train: {len(train_df)} | dev: {len(dev_df)} | test: {len(test_df)}")

if len(train_df) == 0:
    logger.error("No training data found. Check the 'split' column values.")
    sys.exit(1)

# ------------------------------------------------------------------
# Stratified split of train into new_train (80%) and custom_dev (20%)
# ------------------------------------------------------------------
train_df["strat_key"] = train_df.apply(
    lambda row: f"{row['label_sexist']}_{row['label_category']}_{row['label_vector']}",
    axis=1
)

value_counts = train_df["strat_key"].value_counts()
rare_keys = value_counts[value_counts < 5].index
train_df["strat_key"] = train_df["strat_key"].apply(
    lambda x: "__RARE__" if x in rare_keys else x
)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
for train_idx, custom_dev_idx in sss.split(train_df, train_df["strat_key"]):
    new_train = train_df.iloc[train_idx].drop(columns=["strat_key"]).copy()
    custom_dev = train_df.iloc[custom_dev_idx].drop(columns=["strat_key"]).copy()

logger.info(f"Stratified split — new_train: {len(new_train)} | custom_dev: {len(custom_dev)}")

# ------------------------------------------------------------------
# Save all splits
# ------------------------------------------------------------------
splits = {
    "train": new_train,
    "custom_dev": custom_dev,
    "official_dev": dev_df,
    "official_test": test_df,
}

for name, subset in splits.items():
    subset.to_csv(OUT_DIR / f"{name}.csv", index=False)
    logger.info(f"Saved {name}.csv : {len(subset)} rows")

# ------------------------------------------------------------------
# Print Task A distributions for sanity check
# ------------------------------------------------------------------
logger.info("Task A label distribution:")
for name, subset in splits.items():
    if "label_sexist" in subset.columns:
        counts = subset["label_sexist"].value_counts().to_dict()
        logger.info(f"  {name:15s} : {counts}")

logger.info("=" * 60)
logger.info("Hold-out splits created successfully.")
logger.info(f"Directory: {OUT_DIR.absolute()}")
logger.info("Recommended usage:")
logger.info("  - Train on    : splits_strategy_a/train.csv")
logger.info("  - Validate on : splits_strategy_a/custom_dev.csv")
logger.info("  - Final test  : splits_strategy_a/official_test.csv  (run ONCE at project end)")
logger.info("=" * 60)
