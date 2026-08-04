#!/usr/bin/env python3
"""
build_llm_fewshot.py  (v3 — CSV-upload workflow, chunked test output)
Emits, per task, ONE few-shot prompt file (prompt_<task>.txt) whose tail instructs the LLM
to classify an uploaded CSV. The blind test set is split into 8 chunk files of 500 rows each
(test_id_text_1.csv ... test_id_text_8.csv).

Changes vs v2:
  * Blind test CSV is split into NUM_CHUNKS files of CHUNK_SIZE rows each.
  * Prompt tail updated to reference the chunked upload workflow.

Usage:
  python src/build_llm_fewshot.py
  python src/build_llm_fewshot.py --examples_split train
"""
import re, json, argparse, sys, math
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

CODE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)")

# -------------------------
# Constants for chunking
# -------------------------
CHUNK_SIZE = 500
NUM_CHUNKS = 8

B_DEFS = {
    "1": "Threats, plans to harm and incitement: threatens, plans, or encourages harm toward women.",
    "2": "Derogation: demeans, dehumanises, or sexually objectifies women (descriptive / aggressive attacks).",
    "3": "Animosity: casual gendered slurs, profanity, insults, immutable-gender-difference claims, stereotypes, backhanded compliments.",
    "4": "Prejudiced discussions: supports or justifies mistreatment of women, individually or as a group.",
}
C_DEFS = {
    "1.1": "Threats: direct threats of harm toward women.",
    "1.2": "Incitement and encouragement of harm.",
    "2.1": "Descriptive attacks on women's appearance / character.",
    "2.2": "Aggressive and emotive attacks on women.",
    "2.3": "Dehumanising attacks & overt sexual objectification.",
    "3.1": "Casual use of gendered slurs, profanities, and insults.",
    "3.2": "Immutable gender differences and gender stereotypes.",
    "3.3": "Backhanded gendered compliments.",
    "4.1": "Supporting mistreatment of individual women.",
    "4.2": "Supporting mistreatment of women as a group.",
}


def parse_code(value):
    if not isinstance(value, str):
        return None
    m = CODE_RE.match(value)
    return m.group(1) if m else None


def code_sort_key(code):
    return tuple(int(p) for p in code.split("."))


def desc_for(code, defs, raw_fallback):
    return defs.get(code) or raw_fallback.get(code, code)


def sample_n(rng, frame, n):
    if len(frame) == 0:
        return frame
    n = min(n, len(frame))
    idx = rng.choice(np.arange(len(frame)), size=n, replace=False)
    return frame.iloc[idx]


def build_system(task, codes, defs, raw_fallback):
    if task == "a":
        return (
            "You are an expert annotator for the Explainable Detection of Online Sexism (EDOS) dataset.\n"
            "Classify each social-media post as exactly one of:\n"
            "- sexist : the post expresses sexism toward women — overt hostility, derogation, threats,\n"
            "  sexual objectification, OR implicit bias, stereotypes, sarcasm, dog-whistles, and\n"
            "  backhanded compliments that demean or subordinate women.\n"
            "- none   : the post is NOT sexist (it may mention women or gender neutrally, discuss\n"
            "  politics or other topics, or use profanity that is not directed as sexism).\n"
            'For each post, answer with exactly one token: "sexist" or "none".'
        )
    head = ("You are an expert annotator for the EDOS dataset. Assign each post to exactly one "
            + ("sexism category" if task == "b" else "fine-grained sexism vector")
            + ', or "none" if the post is NOT sexist.\n')
    lines = [f"{c} = {desc_for(c, defs, raw_fallback)}" for c in codes]
    final = ('For each post, answer with exactly one token: '
             + ", ".join(codes)
             + ', or "none" (none = the post is not sexist).')
    return head + "\n".join(lines) + "\n" + final


def render_txt(system, examples, allowed, num_chunks=NUM_CHUNKS, chunk_size=CHUNK_SIZE):
    out = ["=== SYSTEM ===", system, "", "=== EXAMPLES ==="]
    for i, e in enumerate(examples, 1):
        out += [f"[Example {i}]", f'Text: "{e["text"]}"', f'predicted: {e["label"]}', ""]
    out += [
        "=== NOW CLASSIFY ===",
        "I've uploaded a CSV file which contains an `id` and a `text` field.",
        f"Classify every row. Your prediction should only be one of: {allowed}.",
        "Create a NEW CSV containing ONLY the two columns `id` and `predicted` "
        "(do NOT repeat the `text` column).",
        "Rules:",
        "- First line is the header: id,predicted",
        "- Then exactly one line per input row, in the EXACT same order as the input.",
        "- Each line is: <the row's original id>,<your prediction>",
        "- Output nothing else (no explanations, no extra columns).",
    ]
    return "\n".join(out)


def split_into_chunks(
    df: pd.DataFrame,
    out_dir: Path,
    num_chunks: int = NUM_CHUNKS,
    chunk_size: int = CHUNK_SIZE,
):
    """
    Split df into `num_chunks` files of `chunk_size` rows each.
    Files are named test_id_text_1.csv ... test_id_text_{num_chunks}.csv.
    Each file contains 'id' and 'text' columns.
    """
    total_needed = num_chunks * chunk_size
    if len(df) < total_needed:
        print(
            f"WARNING: DataFrame has {len(df)} rows but {total_needed} are needed "
            f"({num_chunks} x {chunk_size}). Last chunks may be smaller."
        )

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = df.iloc[start:end]

        if len(chunk) == 0:
            print(f"WARNING: Chunk {i + 1}: no rows remaining, skipping.")
            continue

        file_name = f"test_id_text_{i + 1}.csv"
        file_path = out_dir / file_name
        chunk.to_csv(file_path, index=False)
        print(f"[test] wrote {file_name}  ({len(chunk)} rows; columns id,text)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT / "data" / "processed" / "llm_fewshot"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--examples_split", default="dev", choices=["dev", "train"])
    ap.add_argument("--a_sexist", type=int, default=2)
    ap.add_argument("--a_none", type=int, default=2)
    ap.add_argument("--no_balance", action="store_true")
    ap.add_argument("--b_per_class", type=int, default=3)
    ap.add_argument("--b_none", type=int, default=3)
    ap.add_argument("--c_per_class", type=int, default=2)
    ap.add_argument("--c_none", type=int, default=2)
    ap.add_argument("--chunk_size", type=int, default=CHUNK_SIZE,
                    help=f"Rows per chunk file (default: {CHUNK_SIZE}).")
    ap.add_argument("--num_chunks", type=int, default=NUM_CHUNKS,
                    help=f"Number of chunk files to generate (default: {NUM_CHUNKS}).")
    args = ap.parse_args()

    chunk_size = args.chunk_size
    num_chunks = args.num_chunks

    if args.input:
        in_path = Path(args.input)
    else:
        cands = [PROJECT_ROOT / "data" / "raw" / "edos_labelled_aggregated.csv",
                 PROJECT_ROOT / "edos_labelled_aggregated.csv"]
        in_path = next((c for c in cands if c.exists()), None)
        if in_path is None:
            print(f"ERROR: cannot find edos_labelled_aggregated.csv; pass --input. Tried: {cands}")
            sys.exit(1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading {in_path}")

    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)
    for c in ["rewire_id", "text", "label_sexist", "label_category", "label_vector", "split"]:
        if c not in df.columns:
            print(f"ERROR: missing column {c}; got {df.columns.tolist()}"); sys.exit(1)
    df["split"] = df["split"].str.strip().str.lower()
    df["label_sexist"] = df["label_sexist"].str.strip().str.lower()
    df["label_category"] = df["label_category"].str.strip()
    df["label_vector"] = df["label_vector"].str.strip()

    # ---- blind test set: id + text only, split into chunks ----
    test = df[df["split"] == "test"][["rewire_id", "text"]].rename(columns={"rewire_id": "id"})
    test = test.reset_index(drop=True)
    print(f"[test] total test rows: {len(test)}")
    print(f"[test] splitting into {num_chunks} files of {chunk_size} rows each...")
    split_into_chunks(test, out_dir, num_chunks=num_chunks, chunk_size=chunk_size)

    # ---- label inventories from data ----
    raw_b, raw_c = {}, {}
    for _, r in df.iterrows():
        cb, cc = parse_code(r["label_category"]), parse_code(r["label_vector"])
        if cb and cb not in raw_b:
            raw_b[cb] = r["label_category"][len(cb):].lstrip(". ")
        if cc and cc not in raw_c:
            raw_c[cc] = r["label_vector"][len(cc):].lstrip(". ")
    b_codes = sorted(raw_b, key=code_sort_key)
    c_codes = sorted(raw_c, key=code_sort_key)
    print(f"Task B categories found: {b_codes}")
    print(f"Task C vectors    found: {c_codes}  (count={len(c_codes)})")

    pool = df[df["split"] == args.examples_split].copy()
    pool_sexist = pool[pool["label_sexist"] == "sexist"]
    pool_none = pool[pool["label_sexist"] == "not sexist"]
    pool_b = {c: pool_sexist[pool_sexist["label_category"].map(parse_code) == c] for c in b_codes}
    pool_c = {c: pool_sexist[pool_sexist["label_vector"].map(parse_code) == c] for c in c_codes}
    rng = np.random.default_rng(args.seed)

    def to_examples(frame, label_fn):
        return [{"id": r["rewire_id"], "text": r["text"], "label": label_fn(r)} for _, r in frame.iterrows()]

    # ---- Task A ----
    if args.no_balance:
        a_ex = to_examples(sample_n(rng, pool, args.a_sexist + args.a_none),
                           lambda r: "sexist" if r["label_sexist"] == "sexist" else "none")
    else:
        a_ex = (to_examples(sample_n(rng, pool_sexist, args.a_sexist), lambda r: "sexist")
                + to_examples(sample_n(rng, pool_none, args.a_none), lambda r: "none"))
    rng.shuffle(a_ex)

    # ---- Task B ----
    b_ex = []
    for c in b_codes:
        b_ex += to_examples(sample_n(rng, pool_b[c], args.b_per_class), lambda r, c=c: c)
    b_ex += to_examples(sample_n(rng, pool_none, args.b_none), lambda r: "none")
    rng.shuffle(b_ex)

    # ---- Task C ----
    c_ex = []
    for c in c_codes:
        c_ex += to_examples(sample_n(rng, pool_c[c], args.c_per_class), lambda r, c=c: c)
    c_ex += to_examples(sample_n(rng, pool_none, args.c_none), lambda r: "none")
    rng.shuffle(c_ex)

    allowed = {
        "a": '"sexist" or "none"',
        "b": '"' + '", "'.join(b_codes) + '", or "none"',
        "c": '"' + '", "'.join(c_codes) + '", or "none"',
    }
    tasks = {
        "a": (build_system("a", None, None, None), a_ex),
        "b": (build_system("b", b_codes, B_DEFS, raw_b), b_ex),
        "c": (build_system("c", c_codes, C_DEFS, raw_c), c_ex),
    }

    for task, (system, examples) in tasks.items():
        (out_dir / f"prompt_{task}.txt").write_text(
            render_txt(system, examples, allowed[task], num_chunks=num_chunks, chunk_size=chunk_size),
            encoding="utf-8",
        )
        n_none = sum(1 for e in examples if e["label"] == "none")
        print(f"[task {task}] wrote prompt_{task}.txt  (examples={len(examples)}, none={n_none})")

    print(f"\nDone. Outputs in: {out_dir}")
    print(f"Upload prompt_<task>.txt together with test_id_text_1.csv ... "
          f"test_id_text_{num_chunks}.csv to your LLM;")
    print("it should return each CSV with an added `predicted` column.")


if __name__ == "__main__":
    main()