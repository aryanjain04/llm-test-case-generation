"""
Kaggle Training Notebook for CodeT5 Test Case Generation
=========================================================
Upload this as a Kaggle notebook with GPU T4 x2 accelerator.

SETUP INSTRUCTIONS (read carefully):
1. Go to kaggle.com → New Notebook
2. Upload 'datasets/train_combined.jsonl' as a Kaggle Dataset
   (or upload the whole kaggle_package/ folder as a dataset)
3. Set Accelerator to GPU T4 x2 (free tier)
4. Paste this entire script into the notebook
5. Run all cells
6. Download the output checkpoint files when done

The trained model checkpoint will be saved to /kaggle/working/checkpoints/
which you can download from the Output tab.
"""

# %% [markdown]
# # CodeT5 Fine-Tuning for Test Case Generation
# Using LoRA (Parameter-Efficient Fine-Tuning) on T4 GPU

# %% Cell 1: Install dependencies
# !pip install peft accelerate jsonlines transformers sentencepiece --quiet

# %% Cell 2: Imports and Setup
import os
import json
import ast
import re
from pathlib import Path

import torch
import jsonlines
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from tqdm.auto import tqdm

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

device = "cuda" if torch.cuda.is_available() else "cpu"

# %% Cell 3: Configuration
# ============================================================
# CONFIGURE THESE PATHS BASED ON YOUR KAGGLE DATASET LOCATION
# ============================================================

# If you uploaded as a Kaggle dataset named "llm-testgen-data":
DATA_PATH = "/kaggle/input/llm-testgen-data/train_combined.jsonl"

# If the file is not found, try these alternatives:
if not os.path.exists(DATA_PATH):
    # Maybe uploaded directly
    alternatives = [
        "/kaggle/input/train_combined.jsonl",
        "/kaggle/input/llm-testcase-data/train_combined.jsonl",
        "/kaggle/input/testgen-dataset/train_combined.jsonl",
        "train_combined.jsonl",  # uploaded to notebook
    ]
    for alt in alternatives:
        if os.path.exists(alt):
            DATA_PATH = alt
            break

print(f"Data path: {DATA_PATH}")
assert os.path.exists(DATA_PATH), f"Dataset not found! Upload train_combined.jsonl as a Kaggle dataset. Tried: {DATA_PATH}"

# Training config
CONFIG = {
    "model_name": "Salesforce/codet5-base",     # 220M params (upgrade from small)
    "epochs": 15,
    "batch_size": 8,                             # T4 has 16GB, can handle 8
    "gradient_accumulation_steps": 2,            # effective batch = 16
    "learning_rate": 2e-4,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "max_source_length": 512,
    "max_target_length": 512,
    "warmup_steps": 100,
    "eval_split": 0.1,
    "output_dir": "/kaggle/working/checkpoints/codet5-base-finetuned",
}

print(json.dumps(CONFIG, indent=2))


# %% Cell 4: Dataset class
class FunctionTestDataset(Dataset):
    """PyTorch dataset for function → test pairs."""

    def __init__(self, data_path, tokenizer, max_source_length=512, max_target_length=512):
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

        print(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Source: function code with prompt (must match inference prompt!)
        source = f"Generate pytest tests for:\n{ex['function_code']}"
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


# %% Cell 5: Load model and apply LoRA
print(f"Loading {CONFIG['model_name']}...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=CONFIG["lora_rank"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=["q", "v"],  # Attention Q and V projections
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.to(device)


# %% Cell 6: Prepare data loaders
dataset = FunctionTestDataset(
    DATA_PATH, tokenizer,
    CONFIG["max_source_length"],
    CONFIG["max_target_length"],
)

eval_size = max(1, int(len(dataset) * CONFIG["eval_split"]))
train_size = len(dataset) - eval_size
train_dataset, eval_dataset = torch.utils.data.random_split(
    dataset, [train_size, eval_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
eval_loader = DataLoader(eval_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")


# %% Cell 7: Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
total_steps = len(train_loader) * CONFIG["epochs"] // CONFIG["gradient_accumulation_steps"]
scheduler = get_linear_schedule_with_warmup(optimizer, CONFIG["warmup_steps"], total_steps)

output_path = Path(CONFIG["output_dir"])
output_path.mkdir(parents=True, exist_ok=True)

best_eval_loss = float("inf")
training_log = []
scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

print(f"\n{'='*60}")
print(f"Training {CONFIG['model_name']} with LoRA")
print(f"{'='*60}")
print(f"Total steps: {total_steps}")
print(f"Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")

for epoch in range(1, CONFIG["epochs"] + 1):
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
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
                loss = outputs.loss / CONFIG["gradient_accumulation_steps"]

            scaler.scale(loss).backward()

            if step % CONFIG["gradient_accumulation_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / CONFIG["gradient_accumulation_steps"]
            loss.backward()

            if step % CONFIG["gradient_accumulation_steps"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        epoch_loss += loss.item() * CONFIG["gradient_accumulation_steps"]
        pbar.set_postfix({"loss": f"{loss.item() * CONFIG['gradient_accumulation_steps']:.4f}"})

    avg_train_loss = epoch_loss / len(train_loader)

    # Evaluation
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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

    # Save best model
    if avg_eval_loss < best_eval_loss:
        best_eval_loss = avg_eval_loss
        best_dir = output_path / "best"
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        print(f"  ★ New best model! (eval_loss={avg_eval_loss:.4f})")

    # Save every 5 epochs
    if epoch % 5 == 0:
        checkpoint_dir = output_path / f"epoch-{epoch}"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"  Saved checkpoint: epoch-{epoch}")

# Save training log
log_path = output_path / "training_log.json"
with open(log_path, "w") as f:
    json.dump(training_log, f, indent=2)


# %% Cell 8: Quick validation — generate tests for sample functions
print(f"\n{'='*60}")
print("Quick Validation: Generating tests for sample functions")
print(f"{'='*60}")

# Load the best checkpoint
best_dir = output_path / "best"
base_model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])
peft_model = PeftModel.from_pretrained(base_model, str(best_dir))
merged_model = peft_model.merge_and_unload()
merged_model.to(device)
merged_model.eval()

test_functions = [
    '''def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True''',

    '''def factorial(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)''',

    '''def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1''',
]

for func in test_functions:
    func_name = func.split("(")[0].replace("def ", "").strip()
    prompt = f"Generate pytest tests for:\n{func}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = merged_model.generate(
            **inputs,
            max_new_tokens=256,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n--- {func_name} ---")
    print(generated[:500])
    print()


# %% Cell 9: Save final merged model (optional, for easy local loading)
print("Saving merged model for easy loading...")
merged_dir = output_path / "merged"
merged_dir.mkdir(parents=True, exist_ok=True)
merged_model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)
print(f"Merged model saved to {merged_dir}")

print(f"\n{'='*60}")
print("DONE! Download these folders from the Output tab:")
print(f"  1. {output_path / 'best'} (LoRA adapter — small, ~2MB)")
print(f"  2. {output_path / 'merged'} (Full merged model — ~880MB)")
print(f"  3. {output_path / 'training_log.json'} (Training history)")
print(f"{'='*60}")
print(f"\nBest eval loss: {best_eval_loss:.4f}")
