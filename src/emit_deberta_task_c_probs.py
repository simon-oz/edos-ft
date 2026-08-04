#!/usr/bin/env python3
"""emit_deberta_task_c_probs.py — save DeBERTa Task-C softmax probs (N, C) for the voter."""
import argparse, torch, numpy as np, pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROJECT_ROOT / "models" / "ensemble_probs"
OUT.mkdir(parents=True, exist_ok=True)

def probs_for(model, tok, texts, device, bs=32, max_len=256):
    model.eval(); out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], truncation=True, padding="max_length",
                  max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/deberta/task_c_baseline/final",
                    help="Saved fine-tuned Task-C DeBERTa dir (final/ or best checkpoint)")
    ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(a.model_path).to(device)
    for split in ["dev", "test"]:
        df = pd.read_csv(PROC / f"task_c_{split}.csv")
        p = probs_for(model, tok, df["text"].tolist(), device, a.batch_size)
        np.save(OUT / f"deberta_task_c_{split}_probs.npy", p)
        print(f"saved {OUT / f'deberta_task_c_{split}_probs.npy'} shape={p.shape}")

if __name__ == "__main__":
    main()