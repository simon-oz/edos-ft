#!/usr/bin/env python3
"""
parse_edos_data.py
Parse all EDOS raw CSV files and produce cleaned CSVs + task-specific datasets.

Input files (place in data/raw/):
  - edos_labelled_aggregated.csv
  - edos_labelled_individual_annotations.csv
  - gab_1M_unlabelled.csv
  - reddit_1M_unlabelled.csv

Output files (written to data/processed/):
  - edos_aggregated.csv
  - edos_individual.csv
  - unlabeled_gab.csv, unlabeled_reddit.csv, unlabeled_combined_400k.csv
  - task_a_{train,dev,test}.csv
  - task_b_{train,dev,test}.csv
  - task_c_{train,dev,test}.csv

Usage:
  From project root:  python src/parse_edos_data.py
  From src/ folder:   python parse_edos_data.py
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------------
# Auto-detect project root whether run from project root or src/
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.name == "src":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

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
# Helper: read CSV with clear error messages
# ------------------------------------------------------------------
def read_csv_robust(path: Path, required_cols: list, **kwargs):
    """Read a CSV file and validate required columns."""
    if not path.exists():
        logger.error(f"File not found: {path}")
        logger.error(f"  Expected it at: {path.absolute()}")
        logger.error(f"  Please place '{path.name}' in {path.parent}/")
        sys.exit(1)

    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False,
                         on_bad_lines="warn", **kwargs)
    except Exception as e:
        logger.error(f"Failed to parse {path.name}: {e}")
        sys.exit(1)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"{path.name} is missing columns: {missing}")
        logger.error(f"  Found columns: {list(df.columns)}")
        sys.exit(1)

    logger.info(f"Loaded {path.name}: {len(df)} rows, {len(df.columns)} columns")
    return df


# ------------------------------------------------------------------
# 1. Parse aggregated labels (gold standard)
# ------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Step 1/6: Parsing edos_labelled_aggregated.csv")
logger.info("=" * 60)

agg_path = RAW_DIR / "edos_labelled_aggregated.csv"

df_agg = read_csv_robust(
    agg_path,
    required_cols=["rewire_id", "text", "label_sexist",
                   "label_category", "label_vector", "split"]
)

# Clean text and label columns
df_agg["text"] = df_agg["text"].astype(str).str.strip()
for col in ["label_sexist", "label_category", "label_vector", "split"]:
    df_agg[col] = df_agg[col].astype(str).str.strip()

logger.info(f"Total rows : {len(df_agg)}")
logger.info(f"Splits     : {dict(df_agg['split'].value_counts().sort_index())}")
logger.info(f"Sexist     : {(df_agg['label_sexist'] == 'sexist').sum()}")
logger.info(f"Not sexist : {(df_agg['label_sexist'] == 'not sexist').sum()}")

df_agg.to_csv(PROC_DIR / "edos_aggregated.csv", index=False)
logger.info(f"Saved edos_aggregated.csv to {PROC_DIR}")

# ------------------------------------------------------------------
# 2. Parse individual annotations (for inter-annotator analysis)
# ------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Step 2/6: Parsing edos_labelled_individual_annotations.csv")
logger.info("=" * 60)

ind_path = RAW_DIR / "edos_labelled_individual_annotations.csv"

df_ind = read_csv_robust(
    ind_path,
    required_cols=["rewire_id", "text", "annotator",
                   "label_sexist", "label_category", "label_vector", "split"]
)

df_ind["text"] = df_ind["text"].astype(str).str.strip()
for col in ["label_sexist", "label_category", "label_vector", "split", "annotator"]:
    df_ind[col] = df_ind[col].astype(str).str.strip()

logger.info(f"Total annotations : {len(df_ind)}")
logger.info(f"Unique texts      : {df_ind['rewire_id'].nunique()}")
logger.info(f"Annotators        : {df_ind['annotator'].nunique()}")

df_ind.to_csv(PROC_DIR / "edos_individual.csv", index=False)
logger.info(f"Saved edos_individual.csv to {PROC_DIR}")

# ------------------------------------------------------------------
# 3. Parse unlabeled GAB data
# ------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Step 3/6: Parsing gab_1M_unlabelled.csv")
logger.info("=" * 60)

gab_path = RAW_DIR / "gab_1M_unlabelled.csv"

df_gab = read_csv_robust(gab_path, required_cols=["text"])
df_gab["text"] = df_gab["text"].astype(str).str.strip()
df_gab = df_gab[df_gab["text"] != ""].reset_index(drop=True)

logger.info(f"Total posts : {len(df_gab):,}")
df_gab.to_csv(PROC_DIR / "unlabeled_gab.csv", index=False)
logger.info(f"Saved unlabeled_gab.csv to {PROC_DIR}")

# ------------------------------------------------------------------
# 4. Parse unlabeled Reddit data
# ------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Step 4/6: Parsing reddit_1M_unlabelled.csv")
logger.info("=" * 60)

reddit_path = RAW_DIR / "reddit_1M_unlabelled.csv"

df_reddit = read_csv_robust(reddit_path, required_cols=["text"])
df_reddit["text"] = df_reddit["text"].astype(str).str.strip()
df_reddit = df_reddit[df_reddit["text"] != ""].reset_index(drop=True)

logger.info(f"Total posts : {len(df_reddit):,}")
df_reddit.to_csv(PROC_DIR / "unlabeled_reddit.csv", index=False)
logger.info(f"Saved unlabeled_reddit.csv to {PROC_DIR}")

# ------------------------------------------------------------------
# 5. Build task-specific datasets from aggregated gold labels
# ------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Step 5/6: Building task-specific datasets")
logger.info("=" * 60)

# ---- Task A: Binary ----
for split in ["train", "dev", "test"]:
    subset = df_agg[df_agg["split"] == split][["rewire_id", "text", "label_sexist"]].copy()
    subset["label"] = subset["label_sexist"].map({"sexist": 1, "not sexist": 0})
    subset = subset[["rewire_id", "text", "label"]]
    subset.to_csv(PROC_DIR / f"task_a_{split}.csv", index=False)
    n_sexist = subset["label"].sum()
    logger.info(f"task_a_{split} : {len(subset)} rows "
                f"(sexist={int(n_sexist)}, not_sexist={len(subset) - int(n_sexist)})")

# ---- Task B: 4-category (sexist only) ----
# Build unified category mapping from full dataset to ensure consistent codes
all_sexist = df_agg[df_agg["label_sexist"] == "sexist"].copy()
cat_order = sorted(all_sexist["label_category"].unique())
logger.info(f"Task B categories : {cat_order}")

for split in ["train", "dev", "test"]:
    subset = df_agg[
        (df_agg["split"] == split) & (df_agg["label_sexist"] == "sexist")
    ][["rewire_id", "text", "label_category"]].copy()

    subset["label"] = (
        subset["label_category"]
        .astype("category")
        .cat.set_categories(cat_order)
        .cat.codes
    )
    subset = subset[["rewire_id", "text", "label_category", "label"]]
    subset.to_csv(PROC_DIR / f"task_b_{split}.csv", index=False)
    logger.info(f"task_b_{split} : {len(subset)} rows, "
                f"categories={dict(subset['label_category'].value_counts())}")

# ---- Task C: 11-vector (sexist only) ----
vec_order = sorted(all_sexist["label_vector"].unique())
logger.info(f"Task C vectors    : {vec_order}")

for split in ["train", "dev", "test"]:
    subset = df_agg[
        (df_agg["split"] == split) & (df_agg["label_sexist"] == "sexist")
    ][["rewire_id", "text", "label_vector"]].copy()

    subset["label"] = (
        subset["label_vector"]
        .astype("category")
        .cat.set_categories(vec_order)
        .cat.codes
    )
    subset = subset[["rewire_id", "text", "label_vector", "label"]]
    subset.to_csv(PROC_DIR / f"task_c_{split}.csv", index=False)
    logger.info(f"task_c_{split} : {len(subset)} rows, vectors={subset['label_vector'].nunique()}")

# ------------------------------------------------------------------
# 6. Build combined unlabeled sample for MLM pretraining
# ------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Step 6/6: Building combined unlabeled sample for MLM")
logger.info("=" * 60)

SAMPLE_SIZE = 200_000

df_gab_sample = df_gab.sample(
    n=min(SAMPLE_SIZE, len(df_gab)), random_state=42
)
df_reddit_sample = df_reddit.sample(
    n=min(SAMPLE_SIZE, len(df_reddit)), random_state=42
)
df_unlabeled = pd.concat([df_gab_sample, df_reddit_sample], ignore_index=True)
df_unlabeled = df_unlabeled.sample(frac=1, random_state=42).reset_index(drop=True)

df_unlabeled.to_csv(PROC_DIR / "unlabeled_combined_400k.csv", index=False)
logger.info(f"Combined unlabeled : {len(df_unlabeled):,} rows")

logger.info("=" * 60)
logger.info("All data parsed successfully.")
logger.info(f"Output directory : {PROC_DIR.absolute()}")
logger.info("=" * 60)
