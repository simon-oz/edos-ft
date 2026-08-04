#!/usr/bin/env python3
"""train_deberta_task_c.py
Fine-tune DeBERTa-v3-large (or DAPT-adapted DeBERTa) on EDOS Task C
(fine-grained vector classification). Task-C generalization of train_deberta_task_b.py:
  * paths -> task_c_{train,dev,test}.csv
  * target names read from the 'label_vector' column (generic class_<i> fallback)
  * num_classes detected dynamically (EDOS Task C has 10-11 vectors)

Input:  data/processed/task_c_{train,dev,test}.csv
Output: models/deberta/task_c_baseline/ — checkpoints and final model
        logs/train_deberta_task_c_*.log

Usage:
  CUDA_VISIBLE_DEVICES=0 python src/train_deberta_task_c.py \
      --model_path /data/pyworkspace/edos-ft/models/deberta/dapt_mlm/final \
      --batch_size 16 --grad_accum 2 --epochs 10 --lr 1.2e-5
"""
import sys, json, logging, argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, set_seed,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
PROC_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "deberta" / "task_c_baseline"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_deberta_task_c_{timestamp}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")
logger.info(f"Project root: {PROJECT_ROOT.absolute()}")
logger.info(f"Model output: {MODEL_DIR.absolute()}")


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *a, **kw):
        super().__init__(*a, **kw); self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs); logits = outputs.logits
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.model.config.num_labels), labels.view(-1), weight=w)
        return (loss, outputs) if return_outputs else loss


def compute_metrics_standard(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": float((preds == labels).mean()), "f1_macro": float(f1),
            "precision_macro": float(p), "recall_macro": float(r)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MODEL_NAME = args.model_path
    MAX_LENGTH = 256
    SEED = 42
    set_seed(SEED)

    # 1. Load data
    logger.info("=" * 60); logger.info("Loading Task C datasets"); logger.info("=" * 60)
    train_path, dev_path, test_path = (PROC_DIR / "task_c_train.csv",
                                       PROC_DIR / "task_c_dev.csv",
                                       PROC_DIR / "task_c_test.csv")
    for p in [train_path, dev_path, test_path]:
        if not p.exists():
            logger.error(f"Missing: {p}. Run parse_edos_data.py first."); sys.exit(1)
    dataset = load_dataset("csv", data_files={"train": str(train_path),
                                              "validation": str(dev_path),
                                              "test": str(test_path)})
    logger.info(f"Train : {len(dataset['train'])} samples")
    logger.info(f"Validation : {len(dataset['validation'])} samples")
    logger.info(f"Test : {len(dataset['test'])} samples")

    labels_train = np.array(dataset["train"]["label"], dtype=np.int64)
    num_classes = int(labels_train.max()) + 1
    logger.info(f"Detected {num_classes} classes for Task C.")

    # Target names from label_vector column (fallback generic)
    if "label_vector" in dataset["train"].column_names:
        pairs = list(zip(dataset["train"]["label"], dataset["train"]["label_vector"]))
        label_map = {int(l): v for l, v in pairs}
        target_names = [label_map[i] for i in sorted(label_map)]
    else:
        target_names = [f"class_{i}" for i in range(num_classes)]
        logger.warning("'label_vector' column not found; using generic class names.")
    logger.info(f"Target names: {target_names}")

    # Class weights inversely proportional to frequency
    class_counts = np.bincount(labels_train, minlength=num_classes)
    class_weights = torch.tensor(
        [len(labels_train) / (num_classes * c) if c > 0 else 0.0 for c in class_counts],
        dtype=torch.float)
    logger.info(f"Train class distribution: {class_counts.tolist()}")
    logger.info(f"Class weights: {class_weights.tolist()}")

    # 2. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length",
                         max_length=MAX_LENGTH)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 3. Model
    logger.info("=" * 60); logger.info(f"Loading model: {MODEL_NAME}"); logger.info("=" * 60)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=False,
        torch_dtype=torch.float32)
    logger.info(f"Total parameters : {sum(p.numel() for p in model.parameters()):,}")

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR), eval_strategy="epoch", save_strategy="epoch",
        learning_rate=args.lr, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32, num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.grad_accum, weight_decay=0.01,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        warmup_ratio=0.1, logging_steps=50,
        logging_dir=str(PROJECT_ROOT / "logs" / "tensorboard" / "deberta_task_c"),
        report_to=["tensorboard"], seed=SEED, fp16=True,
        dataloader_num_workers=4, remove_unused_columns=True)

    # 5. Trainer
    trainer = WeightedTrainer(
        class_weights=class_weights, model=model, args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer, compute_metrics=compute_metrics_standard,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

    # 6. Train
    logger.info("=" * 60); logger.info("Starting training"); logger.info("=" * 60)
    train_result = trainer.train()
    logger.info("Training complete.")
    logger.info(f"  Final train loss : {train_result.training_loss:.4f}")
    logger.info(f"  Best checkpoint : {trainer.state.best_model_checkpoint}")
    logger.info(f"  Best F1 (dev) : {trainer.state.best_metric:.4f}")

    # 7. Test evaluation
    logger.info("=" * 60); logger.info("Evaluating on TEST set"); logger.info("=" * 60)
    test_results = trainer.evaluate(tokenized_datasets["test"])
    test_preds = np.argmax(trainer.predict(tokenized_datasets["test"]).predictions, axis=-1)
    test_labels = trainer.predict(tokenized_datasets["test"]).label_ids
    logger.info("Test classification report:")
    logger.info("\n" + classification_report(test_labels, test_preds,
                                             target_names=target_names, digits=4, zero_division=0))

    # 8. Save
    trainer.save_model(MODEL_DIR / "final")
    tokenizer.save_pretrained(MODEL_DIR / "final")
    summary = {"model": MODEL_NAME, "task": "Task C - Fine-Grained Sexism Vector Detection",
               "seed": SEED, "max_length": MAX_LENGTH, "num_classes": num_classes,
               "target_names": target_names,
               "train_samples": len(dataset["train"]), "dev_samples": len(dataset["validation"]),
               "test_samples": len(dataset["test"]),
               "best_checkpoint": trainer.state.best_model_checkpoint,
               "best_dev_f1": float(trainer.state.best_metric) if trainer.state.best_metric else None,
               "test_f1_macro": float(test_results.get("eval_f1_macro", 0)),
               "test_accuracy": float(test_results.get("eval_accuracy", 0)),
               "timestamp": timestamp}
    with open(MODEL_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {MODEL_DIR / 'training_summary.json'}")
    logger.info("=" * 60); logger.info("Task C training complete!"); logger.info("=" * 60)