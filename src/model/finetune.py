"""
Fine-tuning CodeT5-small on function → test case generation.

Uses PEFT/LoRA for parameter-efficient fine-tuning, which fits on 4-6GB VRAM.
LoRA only trains ~0.5% of parameters while matching full fine-tuning quality.

Usage:
    python -m src.model.finetune --data datasets/train.jsonl --epochs 5
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from tqdm import tqdm
import jsonlines


class FunctionTestDataset(Dataset):
    """PyTorch dataset for function → test pairs."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_source_length: int = 512,
        max_target_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.examples = []

        with jsonlines.open(data_path) as reader:
            for obj in reader:
                self.examples.append({
                    "function_code": obj["function_code"],
                    "test_code": obj["test_code"],
                    "function_name": obj.get("function_name", "unknown"),
                })

        print(f"Loaded {len(self.examples)} training examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Source: function code with prompt
        source = f"Generate pytest tests for:\n{ex['function_code']}"

        # Target: the test code
        target = ex["test_code"]

        source_enc = self.tokenizer(
            source,
            max_length=self.max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target_enc = self.tokenizer(
            target,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_enc["input_ids"].squeeze(),
            "attention_mask": source_enc["attention_mask"].squeeze(),
            "labels": labels,
        }


def train(
    data_path: str = "datasets/train.jsonl",
    model_name: str = "Salesforce/codet5-small",
    output_dir: str = "checkpoints/codet5-finetuned",
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    warmup_steps: int = 100,
    max_source_length: int = 512,
    max_target_length: int = 512,
    gradient_accumulation_steps: int = 4,
    save_every: int = 1,
    eval_split: float = 0.1,
):
    """
    Fine-tune CodeT5 with LoRA.

    This is designed to run on a 4-6GB GPU with:
    - LoRA: trains only ~0.5M params instead of 60M
    - Gradient accumulation: simulates larger batch sizes
    - Mixed precision: halves memory for activations
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q", "v"],  # attention Q and V projections
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # Dataset
    full_dataset = FunctionTestDataset(
        data_path, tokenizer, max_source_length, max_target_length
    )

    # Train/eval split
    eval_size = max(1, int(len(full_dataset) * eval_split))
    train_size = len(full_dataset) - eval_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, eval_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # Training loop
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_eval_loss = float("inf")
    training_log = []

    print(f"\n=== Training ===")
    print(f"Train examples: {train_size}")
    print(f"Eval examples: {eval_size}")
    print(f"Epochs: {epochs}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Total steps: {total_steps}\n")

    # Enable mixed precision
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / gradient_accumulation_steps

                scaler.scale(loss).backward()

                if step % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    step_count += 1
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()

                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step_count += 1

            epoch_loss += loss.item() * gradient_accumulation_steps
            pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)

        # Evaluation
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                eval_loss += outputs.loss.item()

        avg_eval_loss = eval_loss / max(len(eval_loader), 1)

        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "eval_loss": round(avg_eval_loss, 4),
            "lr": scheduler.get_last_lr()[0],
        }
        training_log.append(log_entry)

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, eval_loss={avg_eval_loss:.4f}")

        # Save checkpoint
        if epoch % save_every == 0 or avg_eval_loss < best_eval_loss:
            checkpoint_dir = output_path / f"epoch-{epoch}"
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"  Saved checkpoint to {checkpoint_dir}")

            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                best_dir = output_path / "best"
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                print(f"  New best model! (eval_loss={avg_eval_loss:.4f})")

    # Save training log
    log_path = output_path / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nTraining complete. Best eval loss: {best_eval_loss:.4f}")
    print(f"Checkpoints saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CodeT5 for test generation")
    parser.add_argument("--data", default="datasets/train.jsonl", help="Training data path")
    parser.add_argument("--model", default="Salesforce/codet5-small", help="Base model")
    parser.add_argument("--output", default="checkpoints/codet5-finetuned", help="Output dir")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    train(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
    )
