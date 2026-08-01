#!/usr/bin/env python3
"""
train_qwen_task_a.py
Fine-tune Qwen2.5-14B-Instruct on EDOS Task A (binary sexism detection).
Uses likelihood scoring (no generation at eval time) for a calibrated P(sexist).
Structurally parallel to train_qwen_task_b.py (same pipeline; binary vs 4-class).

Modes:
  1. QLoRA 4-bit  (default, ~35 GB VRAM)
  2. LoRA 16-bit  (--use_16bit_lora, ~55 GB VRAM)
  3. Full FT      (--full_finetune --deepspeed_config src/ds_config_zero2.json)

Input:
  data/processed/task_a_{train,dev,test}.csv   (columns: text, label  [0/1])
Output:
  models/qwen/task_a/                          — adapters or full model checkpoints
  models/ensemble_probs/qwen_{dev,test}_probs.npy  — P(sexist), shape (N,)
  logs/train_qwen_task_a_*.log

Usage:
  CUDA_VISIBLE_DEVICES=0 python src/train_qwen_task_a.py
  CUDA_VISIBLE_DEVICES=0 python src/train_qwen_task_a.py --use_16bit_lora

  CUDA_VISIBLE_DEVICES=0 python src/train_qwen_task_a.py \
    --lr 1e-4 --lora_r 16 --lora_alpha 32 --epochs 3
    
  CUDA_VISIBLE_DEVICES=0 python src/train_qwen_task_a.py --use_16bit_lora \
    --lr 5e-5 --lora_r 16 --lora_alpha 32 --epochs 3

  # verify this build is on disk:
  grep -nE "qwen_dev_probs.npy|likelihood_probs" src/train_qwen_task_a.py
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
MODEL_DIR = PROJECT_ROOT / "models" / "qwen" / "task_a"
PROBS_DIR = PROJECT_ROOT / "models" / "ensemble_probs"
MODEL_DIR.mkdir(parents=True, exist_ok=True); PROBS_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_qwen_task_a_{timestamp}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")
logger.info(f"Project root: {PROJECT_ROOT.absolute()}")

# ------------------------------------------------------------------ Task A definitions
SYSTEM_PROMPT = """You are an expert annotator for the Explainable Detection of Online Sexism (EDOS) dataset.
Your task is to classify whether a given text is sexist or not sexist.
Definitions:
- Sexist: The text expresses sexism, including overt hostility, implicit bias, stereotypes, or objectification toward women. This includes sarcasm, dog-whistles, backhanded compliments, and seemingly neutral statements that reinforce gender stereotypes.
- Not Sexist: The text does not express sexism. It may discuss gender-related topics neutrally, or be completely unrelated.
Respond with ONLY a valid JSON object in this exact format:
{"classification": "sexist"}
or
{"classification": "not_sexist"}"""

# The two completions the model is SFT-trained to emit (one per class).
_COMPLETIONS = {
    1: json.dumps({"classification": "sexist"}),
    0: json.dumps({"classification": "not_sexist"}),
}


def format_chat_example(text: str, label: int) -> Dict[str, List[Dict]]:
    """Format a single Task A example into Qwen chat messages. label is 0/1."""
    answer = "sexist" if label == 1 else "not_sexist"
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Text: "{text}"'},
        {"role": "assistant", "content": json.dumps({"classification": answer})},
    ]}


# ------------------------------------------------------------------ SFT data collator (label masking)
class TaskADataCollator:
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


# ------------------------------------------------------------------ likelihood scoring -> P(sexist)
def _build_prompt(tokenizer, text: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Text: "{text}"'}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def likelihood_probs(model, tokenizer, texts, device, batch_size=8, max_length=512):
    """Return P(sexist) for every text, shape (N,). For each class c in {0,1} we build
    [prompt | completion_c], read the model's log-prob of each completion token,
    length-normalise, then 2-way softmax. Requires left padding (set at tokenizer load)."""
    model.eval()
    prev_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    prompts = [_build_prompt(tokenizer, t) for t in texts]
    score = {}
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
                lp = token_logprobs.sum(1) / C
                per.extend(lp.float().cpu().tolist())
            score[cls] = np.array(per, dtype=np.float64)
    finally:
        tokenizer.truncation_side = prev_side
    s1, s0 = score[1], score[0]
    m = np.maximum(s1, s0)
    e1, e0 = np.exp(s1 - m), np.exp(s0 - m)
    return e1 / (e1 + e0)                          # P(sexist), shape (N,)


# ------------------------------------------------------------------ args
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen on EDOS Task A (binary)")
    p.add_argument("--model_name", type=str, default="/data/models/qwen/qwen2.5-14b-it")
    p.add_argument("--use_16bit_lora", action="store_true")
    p.add_argument("--full_finetune", action="store_true")
    p.add_argument("--deepspeed_config", type=str, default=None)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score_batch_size", type=int, default=8)
    return p.parse_args()


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(PROJECT_ROOT / "logs" / "tensorboard" / "qwen_task_a")
    logger.info("=" * 60); logger.info("Qwen Task A Fine-Tuning (binary)"); logger.info("=" * 60)
    logger.info(f"Model  : {args.model_name}")
    logger.info(f"Mode   : {'Full FT' if args.full_finetune else ('16-bit LoRA' if args.use_16bit_lora else '4-bit QLoRA')}")
    logger.info(f"Batch/accum={args.batch_size}/{args.grad_accum} (eff {args.batch_size*args.grad_accum})  "
                f"lr={args.lr}  epochs={args.epochs}  LoRA r/a={args.lora_r}/{args.lora_alpha}")

    # 1. data (standard Task A splits) ------------------------------------------
    logger.info("=" * 60); logger.info("Loading Task A datasets"); logger.info("=" * 60)
    train_path, dev_path, test_path = (PROC_DIR / "task_a_train.csv",
                                       PROC_DIR / "task_a_dev.csv",
                                       PROC_DIR / "task_a_test.csv")
    for pth in (train_path, dev_path, test_path):
        if not pth.exists():
            logger.error(f"Missing: {pth}"); sys.exit(1)
    df_train = pd.read_csv(train_path); df_dev = pd.read_csv(dev_path); df_test = pd.read_csv(test_path)
    df_train["label"] = df_train["label"].astype(int)
    df_dev["label"] = df_dev["label"].astype(int)
    df_test["label"] = df_test["label"].astype(int)
    logger.info(f"Train={len(df_train)}  Dev={len(df_dev)}  Test={len(df_test)}  "
                f"sexist={int(df_train['label'].sum())}")

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
                      processing_class=tokenizer, data_collator=TaskADataCollator(tokenizer, args.max_length),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

    # 7. train ------------------------------------------------------------------
    logger.info("=" * 60); logger.info("Starting training"); logger.info("=" * 60)
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    logger.info(f"Training done in {secs:.0f}s. best dev eval_loss @ {trainer.state.best_model_checkpoint}")

    # 8. likelihood eval + threshold tune + emit probs --------------------------
    logger.info("=" * 60); logger.info("Likelihood scoring on DEV + TEST (binary)"); logger.info("=" * 60)
    dev_texts = df_dev["text"].tolist();  dev_labels = df_dev["label"].to_numpy()
    test_texts = df_test["text"].tolist(); test_labels = df_test["label"].to_numpy()
    dev_probs = likelihood_probs(model, tokenizer, dev_texts, scoring_device,
                                 batch_size=args.score_batch_size, max_length=args.max_length)
    test_probs = likelihood_probs(model, tokenizer, test_texts, scoring_device,
                                  batch_size=args.score_batch_size, max_length=args.max_length)

    # threshold tuned on the clean dev (binary-specific; Task B uses argmax instead)
    # =====Never save corrupted / collapsed probs into the voter =====
    if np.isnan(dev_probs).any() or np.isnan(test_probs).any():
        logger.error("Likelihood probs contain NaN -> weights corrupted (overfit/underflow). "
                     "NOT saving probs or model. Re-run with lower --lr/--lora_r or 4-bit QLoRA.")
        sys.exit(2)
    if float(dev_probs.std()) < 1e-6:
        logger.error(f"Dev probs collapsed (std={float(dev_probs.std()):.2e}) -> constant output. "
                     f"NOT saving probs or model. Re-run with lower --lr/--lora_r or 4-bit QLoRA.")
        sys.exit(2)
    logger.info(f"[guard OK] dev_probs range=[{dev_probs.min():.3f}, {dev_probs.max():.3f}] "
                f"std={float(dev_probs.std()):.3f}")
    # ===== end =====    

    grid = np.arange(0.05, 0.95, 0.005)
    f1s = [f1_score(dev_labels, (dev_probs >= t).astype(int), average="macro", zero_division=0) for t in grid]
    best_t = float(grid[int(np.argmax(f1s))]); best_dev_f1 = float(max(f1s))
    f1_half = float(f1_score(dev_labels, (dev_probs >= 0.5).astype(int), average="macro", zero_division=0))
    logger.info(f"Qwen dev f1_macro @0.50        = {f1_half:.4f}")
    logger.info(f"Qwen dev f1_macro @{best_t:.3f} (tuned) = {best_dev_f1:.4f}")

    test_pred = (test_probs >= best_t).astype(int)
    pc = f1_score(test_labels, test_pred, average=None, zero_division=0)
    test_f1 = float(f1_score(test_labels, test_pred, average="macro", zero_division=0))
    logger.info("Qwen TEST (final look, tuned threshold):")
    logger.info("\n" + classification_report(test_labels, test_pred,
                                             target_names=["not_sexist", "sexist"], digits=4))

    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROBS_DIR / "qwen_dev_probs.npy", dev_probs)
    np.save(PROBS_DIR / "qwen_test_probs.npy", test_probs)
    logger.info(f"Saved qwen probs -> {PROBS_DIR}  (qwen_dev_probs.npy shape={dev_probs.shape}, "
                f"qwen_test_probs.npy shape={test_probs.shape})")

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
        "task": "Task A - Binary Sexism Detection",
        "mode": "full_ft" if args.full_finetune else ("16bit_lora" if args.use_16bit_lora else "4bit_qlora"),
        "seed": args.seed, "max_length": args.max_length,
        "lora_r": args.lora_r if not args.full_finetune else None,
        "lora_alpha": args.lora_alpha if not args.full_finetune else None,
        "train_samples": len(df_train), "dev_samples": len(df_dev), "test_samples": len(df_test),
        "eval_method": "likelihood_scoring", "selection_metric": "eval_loss",
        "optimal_threshold": float(best_t), "dev_f1_macro_tuned": float(best_dev_f1),
        "dev_f1_macro_0.5": float(f1_half), "test_f1_macro": float(test_f1),
        "test_f1_sexist": float(pc[1]) if len(pc) > 1 else 0.0,
        "test_f1_not_sexist": float(pc[0]) if len(pc) > 0 else 0.0,
        "probs_dir": str(PROBS_DIR), "train_seconds": round(secs, 1), "timestamp": timestamp,
    }
    json.dump(summary, open(MODEL_DIR / "training_summary.json", "w"), indent=2)
    logger.info(f"Summary -> {MODEL_DIR/'training_summary.json'}")
    logger.info("=" * 60); logger.info("Qwen Task A complete - probs emitted."); logger.info("=" * 60)


if __name__ == "__main__":
    main()