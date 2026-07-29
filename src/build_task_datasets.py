#!/usr/bin/env python3
"""
build_task_datasets.py
Build Task A/B/C datasets from any split directory (e.g. splits_strategy_a/).

Usage:
  From project root:
    python src/build_task_datasets.py \
        --input_dir data/processed/splits_strategy_a \
        --output_dir data/processed/task_datasets_a

  From src/ folder:
    python build_task_datasets.py \
        --input_dir ../data/processed/splits_strategy_a \
        --output_dir ../data/processed/task_datasets_a
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
import logging
from datetime import datetime


def build_task_a(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: sexist (1) vs not sexist (0). Includes ALL rows."""
    out = df[["rewire_id", "text", "label_sexist"]].copy()
    out["label"] = out["label_sexist"].map({"sexist": 1, "not sexist": 0})
    if out["label"].isna().any():
        bad = out[out["label"].isna()]["label_sexist"].unique()
        logger.warning(f"Task A: unexpected label_sexist values: {bad}")
    out["label"] = out["label"].fillna(-1).astype(int)
    return out[["rewire_id", "text", "label"]]


def build_task_b(df: pd.DataFrame, cat_order: list = None) -> pd.DataFrame:
    """4-category classification (sexist only)."""
    subset = df[df["label_sexist"] == "sexist"].copy()
    if len(subset) == 0:
        return pd.DataFrame(columns=["rewire_id", "text", "label_category", "label"])

    out = subset[["rewire_id", "text", "label_category"]].copy()

    if cat_order is None:
        cat_order = sorted(out["label_category"].unique())

    out["label"] = (
        out["label_category"]
        .astype("category")
        .cat.set_categories(cat_order)
        .cat.codes
    )
    return out[["rewire_id", "text", "label_category", "label"]]


def build_task_c(df: pd.DataFrame, vec_order: list = None) -> pd.DataFrame:
    """11-vector classification (sexist only)."""
    subset = df[df["label_sexist"] == "sexist"].copy()
    if len(subset) == 0:
        return pd.DataFrame(columns=["rewire_id", "text", "label_vector", "label"])

    out = subset[["rewire_id", "text", "label_vector"]].copy()

    if vec_order is None:
        vec_order = sorted(out["label_vector"].unique())

    out["label"] = (
        out["label_vector"]
        .astype("category")
        .cat.set_categories(vec_order)
        .cat.codes
    )
    return out[["rewire_id", "text", "label_vector", "label"]]


def main():
    parser = argparse.ArgumentParser(description="Build Task A/B/C datasets from split CSVs.")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing {train,custom_dev,official_dev,official_test}.csv")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write task_a_{split}.csv, task_b_{split}.csv, task_c_{split}.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Auto-detect project root for logging
    # ------------------------------------------------------------------
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

    LOG_DIR = PROJECT_ROOT / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"build_task_datasets_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    global logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")

    # Derive unified label orders from train.csv
    train_path = input_dir / "train.csv"
    if not train_path.exists():
        logger.error(f"{train_path} not found. Ensure train.csv exists in {input_dir}")
        sys.exit(1)

    train_full = pd.read_csv(train_path, dtype=str, keep_default_na=False)
    for col in ["label_sexist", "label_category", "label_vector"]:
        if col in train_full.columns:
            train_full[col] = train_full[col].str.strip()

    train_sexist = train_full[train_full["label_sexist"] == "sexist"]
    cat_order = sorted(train_sexist["label_category"].unique()) if len(train_sexist) > 0 else []
    vec_order = sorted(train_sexist["label_vector"].unique()) if len(train_sexist) > 0 else []

    logger.info(f"Task B category order (from train): {cat_order}")
    logger.info(f"Task C vector order   (from train): {vec_order}")

    for split_name in ["train", "custom_dev", "official_dev", "official_test"]:
        csv_path = input_dir / f"{split_name}.csv"
        if not csv_path.exists():
            logger.info(f"[SKIP] {csv_path} not found")
            continue

        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        for col in ["label_sexist", "label_category", "label_vector"]:
            if col in df.columns:
                df[col] = df[col].str.strip()

        task_a = build_task_a(df)
        task_a.to_csv(output_dir / f"task_a_{split_name}.csv", index=False)

        task_b = build_task_b(df, cat_order=cat_order)
        task_b.to_csv(output_dir / f"task_b_{split_name}.csv", index=False)

        task_c = build_task_c(df, vec_order=vec_order)
        task_c.to_csv(output_dir / f"task_c_{split_name}.csv", index=False)

        logger.info(f"[✓] {split_name:15s} : A={len(task_a)}, B={len(task_b)}, C={len(task_c)}")

    logger.info(f"All task datasets written to {output_dir.absolute()}")


if __name__ == "__main__":
    main()
