
#!/usr/bin/env python3
"""
emit_qwen_probs.py
Load the trained Qwen LoRA adapter and emit likelihood-scored P(sexist) vectors
for the voter. Run this ONCE after training if the training script didn't emit probs.

Usage:
  CUDA_VISIBLE_DEVICES=0 python src/emit_qwen_probs.py
  CUDA_VISIBLE_DEVICES=0 python src/emit_qwen_probs.py --adapter_dir models/qwen/task_a/final_adapter
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
PROC_DIR = PROJECT_ROOT / "data" / "processed"
PROBS_DIR = PROJECT_ROOT / "models" / "ensemble_probs"
PROBS_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are an expert annotator for the Explainable Detection of Online Sexism (EDOS) dataset.
Your task is to classify whether a given text is sexist or not sexist.
Definitions:
- Sexist: The text expresses sexism, including overt hostility, implicit bias, stereotypes, or objectification toward women. This includes sarcasm, dog-whistles, backhanded compliments, and seemingly neutral statements that reinforce gender stereotypes.
- Not Sexist: The text does not express sexism. It may discuss gender-related topics neutrally, or be completely unrelated.
Respond with ONLY a valid JSON object in this exact format:
{"classification": "sexist"}
or
{"classification": "not_sexist"}"""

_COMPLETIONS = {
    1: json.dumps({"classification": "sexist"}),
    0: json.dumps({"classification": "not_sexist"}),
}


def _build_prompt(tokenizer, text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Text: "{text}"'}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def likelihood_probs(model, tokenizer, texts, device, batch_size=8, max_length=512):
    """P(sexist) per text, shape (N,). Same function as in train_qwen_task_a.py."""
    model.eval()
    prev_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    prompts = [_build_prompt(tokenizer, t) for t in texts]
    score = {}
    try:
        for cls, comp in _COMPLETIONS.items():
            comp_ids = tokenizer.encode(comp, add_special_tokens=False)
            C = len(comp_ids)
            comp_t = torch.tensor([comp_ids], device=device)
            per = []
            for i in range(0, len(prompts), batch_size):
                enc = tokenizer(prompts[i:i + batch_size], add_special_tokens=False,
                                padding=True, truncation=True, max_length=max_length - C,
                                return_tensors="pt")
                p_ids = enc["input_ids"].to(device)
                p_mask = enc["attention_mask"].to(device)
                B, P = p_ids.shape
                ids = torch.cat([p_ids, comp_t.expand(B, -1)], dim=1)
                mask = torch.cat([p_mask, torch.ones(B, C, dtype=p_mask.dtype, device=device)], dim=1)
                with torch.no_grad():
                    logits = model(input_ids=ids, attention_mask=mask).logits
                logp = torch.log_softmax(logits, dim=-1)
                sel = logp[:, P - 1:P - 1 + C, :]
                tok_ids = comp_t.expand(B, -1)
                lp = sel.gather(-1, tok_ids.unsqueeze(-1)).squeeze(-1).sum(1) / C
                per.extend(lp.float().cpu().tolist())
            score[cls] = np.array(per, dtype=np.float64)
    finally:
        tokenizer.truncation_side = prev_side
    s1, s0 = score[1], score[0]
    m = np.maximum(s1, s0)
    e1, e0 = np.exp(s1 - m), np.exp(s0 - m)
    return e1 / (e1 + e0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="/data/models/qwen/qwen2.5-14b-it")
    p.add_argument("--adapter_dir", default="models/qwen/task_a/final_adapter")
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        print(f"ERROR: adapter dir not found: {adapter_dir}"); sys.exit(1)

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    print(f"Loading LoRA adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    device = next(model.parameters()).device
    print(f"Device: {device}")

    # Load standard splits (same CSVs the voter reads labels from)
    df_dev = pd.read_csv(PROC_DIR / "task_a_dev.csv")
    df_test = pd.read_csv(PROC_DIR / "task_a_test.csv")
    df_dev["label"] = df_dev["label"].astype(int)
    df_test["label"] = df_test["label"].astype(int)
    print(f"Dev={len(df_dev)}  Test={len(df_test)}")

    print("Scoring dev...")
    dev_probs = likelihood_probs(model, tokenizer, df_dev["text"].tolist(),
                                 device, batch_size=args.batch_size)
    print("Scoring test...")
    test_probs = likelihood_probs(model, tokenizer, df_test["text"].tolist(),
                                  device, batch_size=args.batch_size)

    np.save(PROBS_DIR / "qwen_dev_probs.npy", dev_probs)
    np.save(PROBS_DIR / "qwen_test_probs.npy", test_probs)
    print(f"Saved: {PROBS_DIR/'qwen_dev_probs.npy'}  shape={dev_probs.shape}")
    print(f"Saved: {PROBS_DIR/'qwen_test_probs.npy'}  shape={test_probs.shape}")

    # Quick sanity: F1 at threshold 0.5
    from sklearn.metrics import f1_score
    dev_pred = (dev_probs >= 0.5).astype(int)
    test_pred = (test_probs >= 0.5).astype(int)
    print(f"Dev  f1_macro @0.5 = {f1_score(df_dev['label'], dev_pred, average='macro', zero_division=0):.4f}")
    print(f"Test f1_macro @0.5 = {f1_score(df_test['label'], test_pred, average='macro', zero_division=0):.4f}")
    print("Done.")


if __name__ == "__main__":
    main()