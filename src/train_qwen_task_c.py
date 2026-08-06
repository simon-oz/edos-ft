#!/usr/bin/env python3
"""
train_qwen_task_c.py
Fine-tune Qwen2.5-14B-Instruct on EDOS Task C (fine-grained vector classification).
Task-C generalization of train_qwen_task_b.py (causal-LM + likelihood scoring):
  * num_classes, vector taxonomy, completions and system prompt are read from the
    data ('label_vector' column) -> works for 10 or 11 vectors, no hard-coding.
  * Completions use the official vector codes, e.g. {"classification": "2.3"}.
  * Likelihood scoring uses SUM of completion log-probs (shared prefix cancels).
  * Surgical recipe from the working Task B build: lr=5e-5, r=16/alpha=32,
    dropout=0.05, wd=0.01, warmup=0.06, label_smoothing=0.1, balanced SFT stream (cap 5x).
  * Optional --calibrate: dev-tuned per-class weights before argmax (off by default).

Input:  data/processed/task_c_{train,dev,test}.csv  (columns: text, label [0-based], label_vector)
Output: models/qwen/task_c/ — adapters / checkpoints
        models/ensemble_probs/qwen_task_c_{dev,test}_probs.npy — (N, C) probs for a voter
        logs/train_qwen_task_c_*.log

Usage:
  CUDA_VISIBLE_DEVICES=0 python src/train_qwen_task_c.py --use_16bit_lora
  CUDA_VISIBLE_DEVICES=0 python src/train_qwen_task_c.py --use_16bit_lora --calibrate
  CUDA_VISIBLE_DEVICES=0 python src/train_qwen_task_c.py --use_16bit_lora \                                        
    --balance_floor 150 --lr 2e-5 --lora_r 32 --lora_alpha 64 --epochs 6

  CUDA_VISIBLE_DEVICES=0 python src/train_qwen_task_c.py --use_16bit_lora \
    --lr 5e-5 --lora_r 16 --lora_alpha 32 --epochs 3 --label_smoothing 0.05    

  CUDA_VISIBLE_DEVICES=0 python src/train_qwen_task_a.py --use_16bit_lora \
    --model_name models/qwen/dapt_lm/final
"""
import sys, json, re, logging, argparse, time
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "qwen" / "task_c"
PROBS_DIR = PROJECT_ROOT / "models" / "ensemble_probs"
MODEL_DIR.mkdir(parents=True, exist_ok=True); PROBS_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_qwen_task_c_{timestamp}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")

CODE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)")


def parse_code(value):
    """'2.3 dehumanising...' -> '2.3'; 'none'/'' -> None."""
    if not isinstance(value, str):
        return None
    m = CODE_RE.match(value)
    return m.group(1) if m else None


def code_sort_key(code):
    return tuple(int(p) for p in code.split("."))


def build_taxonomy(df):
    """{code: description} for every vector present in the data, sorted by code."""
    pairs = {}
    for _, r in df.iterrows():
        code = parse_code(r.get("label_vector", ""))
        if code and code not in pairs:
            pairs[code] = str(r["label_vector"])[len(code):].lstrip(". ")
    return dict(sorted(pairs.items(), key=lambda kv: code_sort_key(kv[0])))


def build_system_prompt(taxonomy):
    lines = "\n".join(f"{code} = {desc}" for code, desc in taxonomy.items())
    example_code = next(iter(taxonomy))
    return ("You are an expert annotator for the Explainable Detection of Online Sexism (EDOS) dataset.\n"
            "Your task is to classify a SEXIST text into exactly one of the following fine-grained vectors:\n"
            + lines + "\n"
            "Respond with ONLY a valid JSON object in this exact format:\n"
            '{"classification": "' + example_code + '"}\n'
            "using the code of the matching vector. No extra text.")


def format_chat_example(text, label, system_prompt, completions) -> Dict[str, List[Dict]]:
    return {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Text: "{text}"'},
        {"role": "assistant", "content": completions[label]},
    ]}


class TaskCDataCollator:
    """Mask everything except the assistant's JSON response (SFT loss = response-only)."""
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer; self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [self.tokenizer.apply_chat_template(it["messages"], tokenize=False,
                                                    add_generation_prompt=False) for it in batch]
        tok = self.tokenizer(texts, max_length=self.max_length, padding=True,
                             truncation=True, return_tensors="pt")
        input_ids = tok["input_ids"]; labels = input_ids.clone()
        marker = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
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


def _build_prompt(tokenizer, text, system_prompt):
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Text: "{text}"'}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def likelihood_probs(model, tokenizer, texts, device, completions, batch_size=8, max_length=512):
    """P(class) per text, shape (N, num_classes). SUM of completion log-probs (no mean):
    the shared prefix cancels in the argmax and the code signal is not attenuated."""
    model.eval()
    prev_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    prompts = [_build_prompt(tokenizer, t, _SP) for t in texts]
    scores = {}
    try:
        for cls in sorted(completions):
            comp_ids = tokenizer.encode(completions[cls], add_special_tokens=False)
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
                lp = token_logprobs.sum(1)                      # SUM, not mean
                per.extend(lp.float().cpu().tolist())
            scores[cls] = np.array(per, dtype=np.float64)
    finally:
        tokenizer.truncation_side = prev_side
    keys = sorted(completions)
    score_matrix = np.stack([scores[k] for k in keys], axis=1)  # (N, C)
    score_matrix -= score_matrix.max(axis=1, keepdims=True)
    exp_scores = np.exp(score_matrix)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen on EDOS Task C (fine-grained vectors)")
    p.add_argument("--model_name", type=str, default="/data/models/qwen/qwen2.5-14b-it")
    p.add_argument("--use_16bit_lora", action="store_true")
    p.add_argument("--full_finetune", action="store_true")
    p.add_argument("--deepspeed_config", type=str, default=None)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--scheduler", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score_batch_size", type=int, default=8)
    p.add_argument("--no_balance", dest="balance", action="store_false")
    p.add_argument("--calibrate", action="store_true",
                   help="Dev-tuned per-class weight calibration before argmax (optional)")
    p.set_defaults(balance=True)
    p.add_argument("--balance_floor", type=int, default=0,
                help="Minimum samples per class after balancing (0=off), e.g. 150")
    return p.parse_args()


def main():
    global _SP
    args = parse_args()
    set_seed(args.seed)
    logger.info("=" * 60); logger.info("Qwen Task C Fine-Tuning (fine-grained vectors)"); logger.info("=" * 60)
    logger.info(f"Model  : {args.model_name}")
    logger.info(f"Mode   : {'Full FT' if args.full_finetune else ('16-bit LoRA' if args.use_16bit_lora else '4-bit QLoRA')}")
    logger.info(f"  lr={args.lr}, wd={args.wd}, epochs={args.epochs}, warmup={args.warmup_ratio}, "
                f"scheduler={args.scheduler}, label_smoothing={args.label_smoothing}")
    logger.info(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

    # 1. data ------------------------------------------------------------------
    logger.info("=" * 60); logger.info("Loading Task C datasets"); logger.info("=" * 60)
    train_path, dev_path, test_path = (PROC_DIR / "task_c_train.csv",
                                       PROC_DIR / "task_c_dev.csv",
                                       PROC_DIR / "task_c_test.csv")
    for pth in (train_path, dev_path, test_path):
        if not pth.exists():
            logger.error(f"Missing: {pth}"); sys.exit(1)
    df_train = pd.read_csv(train_path); df_dev = pd.read_csv(dev_path); df_test = pd.read_csv(test_path)
    for df in (df_train, df_dev, df_test):
        df["label"] = df["label"].astype(int)
    num_classes = int(df_train["label"].max()) + 1
    logger.info(f"Train={len(df_train)}  Dev={len(df_dev)}  Test={len(df_test)}  num_classes={num_classes}")
    logger.info(f"Train class distribution: {dict(df_train['label'].value_counts().sort_index())}")

    # taxonomy + completions + target names, all data-driven
    taxonomy = build_taxonomy(df_train)
    logger.info(f"Detected {len(taxonomy)} vectors: {list(taxonomy.keys())}")
    if "label_vector" in df_train.columns:
        lv = df_train.drop_duplicates("label").set_index("label")["label_vector"].to_dict()
        target_names = [str(lv.get(i, f"class_{i}")) for i in range(num_classes)]
        codes_by_label = {int(l): (parse_code(v) or str(l)) for l, v in lv.items()}
    else:
        target_names = [f"class_{i}" for i in range(num_classes)]
        codes_by_label = {i: str(i) for i in range(num_classes)}
    completions = {i: json.dumps({"classification": codes_by_label.get(i, str(i))})
                   for i in range(num_classes)}
    _SP = build_system_prompt(taxonomy)

    # balance the SFT stream (cap 5x)
    if args.balance:
        counts = df_train["label"].value_counts()
        max_c = int(counts.max()); cap = 5
        parts = []
        for lab in sorted(df_train["label"].unique()):
            sub = df_train[df_train["label"] == lab]
            base = min(max_c, cap * len(sub))
            target = max(base, args.balance_floor) if args.balance_floor > 0 else base
            reps = max(1, int(np.ceil(target / len(sub))))
            parts.append(pd.concat([sub] * reps, ignore_index=True))
        df_train = (pd.concat(parts, ignore_index=True)
                      .sample(frac=1.0, random_state=args.seed).reset_index(drop=True))
        logger.info(f"Balanced train size: {len(df_train)} (per-class cap={cap}x)")

    train_dataset = Dataset.from_list([format_chat_example(r["text"], r["label"], _SP, completions)
                                       for _, r in df_train.iterrows()])
    dev_dataset = Dataset.from_list([format_chat_example(r["text"], r["label"], _SP, completions)
                                     for _, r in df_dev.iterrows()])

    # 2. tokenizer / model -----------------------------------------------------
    logger.info("=" * 60); logger.info(f"Loading tokenizer: {args.model_name}"); logger.info("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("=" * 60); logger.info("Loading model"); logger.info("=" * 60)
    if args.full_finetune:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16,
                                                     device_map="auto", trust_remote_code=True)
        model.gradient_checkpointing_enable()
    elif args.use_16bit_lora:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16,
                                                     device_map="auto", trust_remote_code=True)
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb,
                                                     device_map="auto", trust_remote_code=True)
        model = prepare_model_for_kbit_training(model)

    if not args.full_finetune:
        logger.info("=" * 60); logger.info("Applying LoRA"); logger.info("=" * 60)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type=TaskType.CAUSAL_LM))
        model.print_trainable_parameters()

    scoring_device = next(model.parameters()).device
    logger.info(f"Scoring device: {scoring_device}")

    # 3. training args ---------------------------------------------------------
    logger.info("=" * 60); logger.info("Configuring training"); logger.info("=" * 60)
    steps_per_ep = max(1, len(df_train) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_ep * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    logger.info(f"Effective batch: {args.batch_size * args.grad_accum}; total steps {total_steps}, warmup {warmup_steps}")
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        weight_decay=args.wd, warmup_steps=warmup_steps, lr_scheduler_type=args.scheduler,
        label_smoothing_factor=args.label_smoothing,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=10, bf16=True, fp16=False,
        dataloader_num_workers=4, remove_unused_columns=False, report_to="none", seed=args.seed,
        deepspeed=args.deepspeed_config if args.full_finetune else None,
    )

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_dataset, eval_dataset=dev_dataset,
                      processing_class=tokenizer, data_collator=TaskCDataCollator(tokenizer, args.max_length),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

    # 4. train -----------------------------------------------------------------
    logger.info("=" * 60); logger.info("Starting training"); logger.info("=" * 60)
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    logger.info(f"Training done in {secs:.0f}s. best dev eval_loss @ {trainer.state.best_model_checkpoint}")

    # 5. likelihood scoring on DEV + TEST --------------------------------------
    logger.info("=" * 60); logger.info(f"Likelihood scoring on DEV + TEST ({num_classes}-class)"); logger.info("=" * 60)
    dev_labels = df_dev["label"].to_numpy(); test_labels = df_test["label"].to_numpy()
    dev_probs = likelihood_probs(model, tokenizer, df_dev["text"].tolist(),
                                 scoring_device, completions, args.score_batch_size, args.max_length)
    test_probs = likelihood_probs(model, tokenizer, df_test["text"].tolist(),
                                  scoring_device, completions, args.score_batch_size, args.max_length)

    dev_pred = dev_probs.argmax(axis=1); test_pred = test_probs.argmax(axis=1)
    dev_f1 = float(f1_score(dev_labels, dev_pred, average="macro", zero_division=0))
    test_f1 = float(f1_score(test_labels, test_pred, average="macro", zero_division=0))
    logger.info(f"Qwen DEV  f1_macro (argmax) = {dev_f1:.4f}")
    logger.info(f"Qwen TEST f1_macro (argmax) = {test_f1:.4f}")
    logger.info("Qwen TEST classification report (raw argmax):")
    logger.info("\n" + classification_report(test_labels, test_pred, target_names=target_names,
                                             digits=4, zero_division=0))

    cal_test_f1 = None
    if args.calibrate:
        from scipy.optimize import differential_evolution
        def neg_macro_f1(logw):
            w = np.exp(logw - logw.max())
            return -f1_score(dev_labels, (dev_probs * w).argmax(axis=1), average="macro", zero_division=0)
        res = differential_evolution(neg_macro_f1, [(-2, 2)] * num_classes, seed=args.seed,
                                     popsize=15, maxiter=60)
        cal_w = np.exp(res.x - res.x.max())
        logger.info(f"Per-class calibration weights: {np.round(cal_w, 3).tolist()}")
        dev_pred = (dev_probs * cal_w).argmax(axis=1)
        test_pred = (test_probs * cal_w).argmax(axis=1)
        logger.info(f"Qwen DEV  f1_macro (calibrated) = {float(f1_score(dev_labels, dev_pred, average='macro', zero_division=0)):.4f}")
        cal_test_f1 = float(f1_score(test_labels, test_pred, average="macro", zero_division=0))
        logger.info(f"Qwen TEST f1_macro (calibrated) = {cal_test_f1:.4f}")
        logger.info("Qwen TEST classification report (calibrated):")
        logger.info("\n" + classification_report(test_labels, test_pred, target_names=target_names,
                                                 digits=4, zero_division=0))

    # 6. save probs + model + summary ------------------------------------------
    np.save(PROBS_DIR / "qwen_task_c_dev_probs.npy", dev_probs)
    np.save(PROBS_DIR / "qwen_task_c_test_probs.npy", test_probs)
    logger.info(f"Saved qwen Task C probs -> {PROBS_DIR}  "
                f"(dev shape={dev_probs.shape}, test shape={test_probs.shape})")

    logger.info("=" * 60); logger.info("Saving model"); logger.info("=" * 60)
    if args.full_finetune:
        trainer.save_model(str(MODEL_DIR / "final"))
    else:
        model.save_pretrained(str(MODEL_DIR / "final_adapter"))
        tokenizer.save_pretrained(str(MODEL_DIR / "final_adapter"))
        tokenizer.save_pretrained(str(MODEL_DIR / "final"))
    summary = {"model": args.model_name, "task": "Task C - Fine-Grained Sexism Vector Detection",
               "mode": "full_ft" if args.full_finetune else ("16bit_lora" if args.use_16bit_lora else "4bit_qlora"),
               "seed": args.seed, "max_length": args.max_length, "num_classes": num_classes,
               "vectors": list(taxonomy.keys()),
               "lr": args.lr, "wd": args.wd, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
               "lora_dropout": args.lora_dropout, "label_smoothing": args.label_smoothing,
               "calibrate": args.calibrate,
               "train_samples": len(df_train), "dev_samples": len(df_dev), "test_samples": len(df_test),
               "eval_method": "likelihood_scoring", "selection_metric": "eval_loss",
               "dev_f1_macro": dev_f1, "test_f1_macro": test_f1,
               "test_f1_macro_calibrated": cal_test_f1,
               "train_seconds": round(secs, 1), "timestamp": timestamp}
    json.dump(summary, open(MODEL_DIR / "training_summary.json", "w"), indent=2)
    logger.info(f"Summary -> {MODEL_DIR / 'training_summary.json'}")
    logger.info("=" * 60); logger.info("Qwen Task C complete - probs emitted."); logger.info("=" * 60)


if __name__ == "__main__":
    main()