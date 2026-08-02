#!/usr/bin/env python3
"""
train_qwen_task_b.py
Fine-tune Qwen2.5-14B-Instruct on EDOS Task B (4-category sexism classification).
Mirrors the structure of the working train_deberta_task_b.py but uses a causal-LM
pipeline with likelihood scoring (no generation at eval time).

Task B categories (EDOS):
  1. threats, plans to harm and incitement
  2. derogation
  3. animosity
  4. prejudiced discussions

Modes:
  1. QLoRA 4-bit  (default, ~35 GB VRAM)
  2. LoRA 16-bit  (--use_16bit_lora, ~55 GB VRAM)
  3. Full FT      (--full_finetune --deepspeed_config src/ds_config_zero2.json)

Input:
  data/processed/task_b_{train,dev,test}.csv   (columns: text, label  [0-3])
Output:
  models/qwen/task_b/                          — adapters or full model checkpoints
  models/ensemble_probs/qwen_task_b_{dev,test}_probs.npy  — (N, 4) probs for a voter
  logs/train_qwen_task_b_*.log

Usage:
  CUDA_VISIBLE_DEVICES=0 python src/train_qwen_task_b.py
  CUDA_VISIBLE_DEVICES=0 python src/train_qwen_task_b.py --use_16bit_lora
  # verify this build is on disk:
  grep -nE "qwen_task_b_dev_probs.npy|likelihood_probs" src/train_qwen_task_b.py
"""
import sys, json, logging, argparse, os, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, EarlyStoppingCallback, set_seed, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# ------------------------------------------------------------------ paths / logging
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "qwen" / "task_b"
PROBS_DIR = PROJECT_ROOT / "models" / "ensemble_probs"
MODEL_DIR.mkdir(parents=True, exist_ok=True); PROBS_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_qwen_task_b_{timestamp}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")
logger.info(f"Project root: {PROJECT_ROOT.absolute()}")

# ------------------------------------------------------------------ Task B definitions
NUM_CLASSES = 4
CATEGORY_NAMES = {
    0: "1. threats, plans to harm and incitement",
    1: "2. derogation",
    2: "3. animosity",
    3: "4. prejudiced discussions",
}
CATEGORY_SHORT = {0: "1", 1: "2", 2: "3", 3: "4"}

SYSTEM_PROMPT = """You are an expert annotator for the Explainable Detection of Online Sexism (EDOS) dataset.
Your task is to classify a sexist text into one of four categories:
1. Threats, plans to harm and incitement: Text that threatens, plans, or incites harm against women.
2. Derogation: Text that demeans, dehumanises, or sexually objectifies women.
3. Animosity: Text that expresses animosity toward women, including casual slurs, profanities, insults, immutable gender differences, stereotypes, and backhanded compliments.
4. Prejudiced discussions: Text that supports or justifies mistreatment of women, either individually or as a group.
Respond with ONLY a valid JSON object in this exact format:
{"classification": "1"}
or
{"classification": "2"}
or
{"classification": "3"}
or
{"classification": "4"}"""

# The four completions the model is SFT-trained to emit (one per class, 0-indexed).
_COMPLETIONS = {
    0: json.dumps({"classification": "1"}),   # threats
    1: json.dumps({"classification": "2"}),   # derogation
    2: json.dumps({"classification": "3"}),   # animosity
    3: json.dumps({"classification": "4"}),   # prejudiced discussions
}


# ------------------------------------------------------------------ formatting
def format_chat_example(text: str, label: int) -> Dict[str, List[Dict]]:
    """Format a single Task B example into Qwen chat messages. label is 0-3."""
    answer = CATEGORY_SHORT[label]
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Text: "{text}"'},
        {"role": "assistant", "content": json.dumps({"classification": answer})},
    ]}


# ------------------------------------------------------------------ SFT data collator (label masking)
class TaskBDataCollator:
    """Mask everything except the assistant's JSON response so SFT loss is response-only."""
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer; self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [self.tokenizer.apply_chat_template(it["messages"], tokenize=False,
                                                    add_generation_prompt=False) for it in batch]
        tok = self.tokenizer(texts, max_length=self.max_length, padding=True,
                             truncation=True, return_tensors="pt")
        input_ids = tok["input_ids"]; labels = input_ids.clone()
        marker = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        if len(marker) < 2:
            marker = self.tokenizer.encode("assistant", add_special_tokens=False)
        for i in range(len(batch)):
            seq = input_ids[i].tolist(); pos = -1
            for j in range(len(seq) - len(marker), -1, -1):
                if seq[j:j + len(marker)] == marker:
                    pos = j; break
            if pos != -1:
                labels[i, :pos + len(marker)] = -100
            else:
                labels[i, :-10] = -100
                logger.warning(f"Could not find assistant marker in batch item {i}")
        tok["labels"] = labels
        return tok


# ------------------------------------------------------------------ likelihood scoring -> P(class)
def _build_prompt(tokenizer, text: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Text: "{text}"'}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def likelihood_probs(model, tokenizer, texts, device, batch_size=8, max_length=512):
    """Return P(class) for every text, shape (N, NUM_CLASSES).
    For each class c we build [prompt | completion_c], read the model's log-prob of each
    completion token, length-normalise, then softmax across all classes."""
    model.eval()
    prev_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    prompts = [_build_prompt(tokenizer, t) for t in texts]
    scores = {}
    try:
        for cls, comp in _COMPLETIONS.items():
            comp_ids = tokenizer.encode(comp, add_special_tokens=False)
            C = len(comp_ids)
            if C == 0:
                raise ValueError(f"Empty completion tokenization for class {cls}")
            comp_t = torch.tensor([comp_ids], device=device)
            per = []
            for i in range(0, len(prompts), batch_size):
                enc = tokenizer(prompts[i:i + batch_size], add_special_tokens=False,
                                padding=True, truncation=True, max_length=max_length - C,
                                return_tensors="pt")
                p_ids = enc["input_ids"].to(device); p_mask = enc["attention_mask"].to(device)
                B, P = p_ids.shape
                ids = torch.cat([p_ids, comp_t.expand(B, -1)], dim=1)
                mask = torch.cat([p_mask, torch.ones(B, C, dtype=p_mask.dtype, device=device)], dim=1)
                with torch.no_grad():
                    logits = model(input_ids=ids, attention_mask=mask).logits
                logp = torch.log_softmax(logits, dim=-1)
                sel = logp[:, P - 1:P - 1 + C, :]
                tok_ids = comp_t.expand(B, -1)
                token_logprobs = sel.gather(-1, tok_ids.unsqueeze(-1)).squeeze(-1)
                lp = token_logprobs.sum(1)
                per.extend(lp.float().cpu().tolist())
            scores[cls] = np.array(per, dtype=np.float64)
    finally:
        tokenizer.truncation_side = prev_side
    # Stack (N, NUM_CLASSES) and stable softmax
    score_matrix = np.stack([scores[c] for c in sorted(scores.keys())], axis=1)
    score_matrix -= score_matrix.max(axis=1, keepdims=True)
    exp_scores = np.exp(score_matrix)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return probs  # (N, NUM_CLASSES)


# ------------------------------------------------------------------ args
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen on EDOS Task B (4-class)")
    p.add_argument("--model_name", type=str, default="/data/models/qwen/qwen2.5-14b-it")
    p.add_argument("--use_16bit_lora", action="store_true")
    p.add_argument("--full_finetune", action="store_true")
    p.add_argument("--deepspeed_config", type=str, default=None)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score_batch_size", type=int, default=8)
    p.add_argument("--no_balance", dest="balance", action="store_false",
                   help="Disable class-balancing of the SFT stream (default: balance)")
    p.set_defaults(balance=True)    
    return p.parse_args()


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(PROJECT_ROOT / "logs" / "tensorboard" / "qwen_task_b")
    logger.info("=" * 60); logger.info("Qwen Task B Fine-Tuning (4-class)"); logger.info("=" * 60)
    logger.info(f"Model  : {args.model_name}")
    logger.info(f"Mode   : {'Full FT' if args.full_finetune else ('16-bit LoRA' if args.use_16bit_lora else '4-bit QLoRA')}")
    logger.info(f"Batch/accum={args.batch_size}/{args.grad_accum} (eff {args.batch_size*args.grad_accum})  "
                f"lr={args.lr}  epochs={args.epochs}  LoRA r/a={args.lora_r}/{args.lora_alpha}")

    # 1. data (standard Task B splits) ------------------------------------------
    logger.info("=" * 60); logger.info("Loading Task B datasets"); logger.info("=" * 60)
    train_path, dev_path, test_path = (PROC_DIR / "task_b_train.csv",
                                       PROC_DIR / "task_b_dev.csv",
                                       PROC_DIR / "task_b_test.csv")
    for pth in (train_path, dev_path, test_path):
        if not pth.exists():
            logger.error(f"Missing: {pth}"); sys.exit(1)
    df_train = pd.read_csv(train_path); df_dev = pd.read_csv(dev_path); df_test = pd.read_csv(test_path)
    df_train["label"] = df_train["label"].astype(int)
    df_dev["label"] = df_dev["label"].astype(int)
    df_test["label"] = df_test["label"].astype(int)
    num_classes = len(np.unique(df_train["label"].values))
    logger.info(f"Train={len(df_train)}  Dev={len(df_dev)}  Test={len(df_test)}  "
                f"num_classes={num_classes}")
    logger.info(f"Train class distribution: {dict(df_train['label'].value_counts().sort_index())}")

    # ===== Balance the SFT stream across the 4 classes =====
    # Without this the 89/454/333/94 skew teaches the model the base-rate, not the
    # conditional digit -> it collapses to the mode. Repeat each class up to the max
    # count, but cap repetition at 5x so tiny classes aren't memorised.
    if args.balance:
        counts = df_train["label"].value_counts()
        max_c = int(counts.max()); cap = 5
        parts = []
        for lab in sorted(df_train["label"].unique()):
            sub = df_train[df_train["label"] == lab]
            target = min(max_c, cap * len(sub))
            reps = max(1, int(np.ceil(target / len(sub))))
            parts.append(pd.concat([sub] * reps, ignore_index=True))
        df_train = (pd.concat(parts, ignore_index=True)
                      .sample(frac=1.0, random_state=args.seed)
                      .reset_index(drop=True))
        logger.info(f"Balanced train size: {len(df_train)} (per-class cap={cap}x)")
        logger.info(f"Balanced distribution: {dict(df_train['label'].value_counts().sort_index())}")

    train_dataset = Dataset.from_list([format_chat_example(r["text"], r["label"]) for _, r in df_train.iterrows()])
    dev_dataset = Dataset.from_list([format_chat_example(r["text"], r["label"]) for _, r in df_dev.iterrows()])

    # 2. tokenizer --------------------------------------------------------------
    logger.info("=" * 60); logger.info(f"Loading tokenizer: {args.model_name}"); logger.info("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 3. model ------------------------------------------------------------------
    logger.info("=" * 60); logger.info("Loading model"); logger.info("=" * 60)
    if args.full_finetune:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16,
                                                     device_map="auto", trust_remote_code=True)
        model.gradient_checkpointing_enable()
    elif args.use_16bit_lora:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16,
                                                     device_map="auto", trust_remote_code=True)
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb,
                                                     device_map="auto", trust_remote_code=True)
        model = prepare_model_for_kbit_training(model)

    # 4. LoRA -------------------------------------------------------------------
    if not args.full_finetune:
        logger.info("=" * 60); logger.info("Applying LoRA"); logger.info("=" * 60)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM))
        model.print_trainable_parameters()

    scoring_device = next(model.parameters()).device
    logger.info(f"Scoring device: {scoring_device}")

    # 5. training args (selection on dev eval_loss; plain Trainer) ---------------
    logger.info("=" * 60); logger.info("Configuring training"); logger.info("=" * 60)
    steps_per_ep = max(1, len(df_train) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_ep * args.epochs
    warmup_steps = int(0.1 * total_steps)
    lr = args.lr if not args.full_finetune else 1e-5
    wd = 0.01 if not args.full_finetune else 0.1
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.grad_accum, learning_rate=lr, weight_decay=wd,
        warmup_steps=warmup_steps, logging_steps=10,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        bf16=True, fp16=False, dataloader_num_workers=4, remove_unused_columns=False,
        report_to=["tensorboard"], seed=args.seed,
        deepspeed=args.deepspeed_config if args.full_finetune else None,
    )
    logger.info(f"lr={lr}  wd={wd}  eff_batch={args.batch_size*args.grad_accum}  "
                f"warmup_steps={warmup_steps}  selection=eval_loss(lower better)")

    # 6. trainer ----------------------------------------------------------------
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_dataset, eval_dataset=dev_dataset,
                      processing_class=tokenizer, data_collator=TaskBDataCollator(tokenizer, args.max_length),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

    # 7. train ------------------------------------------------------------------
    logger.info("=" * 60); logger.info("Starting training"); logger.info("=" * 60)
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    logger.info(f"Training done in {secs:.0f}s. best dev eval_loss @ {trainer.state.best_model_checkpoint}")

    # 8. likelihood eval + emit probs -------------------------------------------
    logger.info("=" * 60); logger.info("Likelihood scoring on DEV + TEST (4-class)"); logger.info("=" * 60)
    dev_texts = df_dev["text"].tolist();  dev_labels = df_dev["label"].to_numpy()
    test_texts = df_test["text"].tolist(); test_labels = df_test["label"].to_numpy()
    dev_probs = likelihood_probs(model, tokenizer, dev_texts, scoring_device,
                                 batch_size=args.score_batch_size, max_length=args.max_length)
    test_probs = likelihood_probs(model, tokenizer, test_texts, scoring_device,
                                  batch_size=args.score_batch_size, max_length=args.max_length)

    dev_pred = dev_probs.argmax(axis=1)
    test_pred = test_probs.argmax(axis=1)
    dev_f1 = float(f1_score(dev_labels, dev_pred, average="macro", zero_division=0))
    test_f1 = float(f1_score(test_labels, test_pred, average="macro", zero_division=0))
    logger.info(f"Qwen DEV  f1_macro (argmax) = {dev_f1:.4f}")
    logger.info(f"Qwen TEST f1_macro (argmax) = {test_f1:.4f}")
    logger.info("Qwen TEST classification report:")
    logger.info("\n" + classification_report(
        test_labels, test_pred,
        target_names=[CATEGORY_NAMES[i] for i in range(num_classes)], digits=4))

    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROBS_DIR / "qwen_task_b_dev_probs.npy", dev_probs)
    np.save(PROBS_DIR / "qwen_task_b_test_probs.npy", test_probs)
    logger.info(f"Saved qwen Task B probs -> {PROBS_DIR}  "
                f"(qwen_task_b_dev_probs.npy shape={dev_probs.shape}, "
                f"qwen_task_b_test_probs.npy shape={test_probs.shape})")

    # 9. save model + summary ---------------------------------------------------
    logger.info("=" * 60); logger.info("Saving model"); logger.info("=" * 60)
    if args.full_finetune:
        trainer.save_model(str(MODEL_DIR / "final"))
    else:
        model.save_pretrained(str(MODEL_DIR / "final_adapter"))
        tokenizer.save_pretrained(str(MODEL_DIR / "final_adapter"))
        tokenizer.save_pretrained(str(MODEL_DIR / "final"))
    summary = {
        "model": args.model_name,
        "task": "Task B - 4-Category Sexism Detection",
        "mode": "full_ft" if args.full_finetune else ("16bit_lora" if args.use_16bit_lora else "4bit_qlora"),
        "seed": args.seed, "max_length": args.max_length, "num_classes": num_classes,
        "lora_r": args.lora_r if not args.full_finetune else None,
        "lora_alpha": args.lora_alpha if not args.full_finetune else None,
        "train_samples": len(df_train), "dev_samples": len(df_dev), "test_samples": len(df_test),
        "eval_method": "likelihood_scoring", "selection_metric": "eval_loss",
        "dev_f1_macro": float(dev_f1), "test_f1_macro": float(test_f1),
        "probs_dir": str(PROBS_DIR), "train_seconds": round(secs, 1), "timestamp": timestamp,
    }
    json.dump(summary, open(MODEL_DIR / "training_summary.json", "w"), indent=2)
    logger.info(f"Summary -> {MODEL_DIR/'training_summary.json'}")
    logger.info("=" * 60); logger.info("Qwen Task B complete - probs emitted."); logger.info("=" * 60)


if __name__ == "__main__":
    main()