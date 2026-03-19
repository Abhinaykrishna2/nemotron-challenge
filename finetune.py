"""
LoRA fine-tuning for NVIDIA Nemotron on the Alice's Wonderland reasoning dataset.

Requirements:
    pip install unsloth transformers datasets trl peft accelerate bitsandbytes

Run on Kaggle / Colab (A100 or 2x A6000 recommended for 30B model).
For smaller GPUs, set MODEL_ID to the 8B variant.

Submission constraint:
    - LoRA rank ≤ 32
    - Outputs: adapter_model.safetensors + adapter_config.json  → zip to submission.zip
"""

import json
import re
import os
from pathlib import Path
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID    = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
OUTPUT_DIR  = Path("outputs/lora_adapter")
SFT_FILE    = Path("data/finetune_sft.jsonl")

LORA_RANK       = 16      # must be ≤ 32 per challenge rules
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
TARGET_MODULES  = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

MAX_SEQ_LEN     = 8192    # match challenge max_model_len
BATCH_SIZE      = 1       # increase if VRAM allows
GRAD_ACCUM      = 8       # effective batch = BATCH_SIZE * GRAD_ACCUM
EPOCHS          = 3
LR              = 2e-4
WARMUP_RATIO    = 0.05
LR_SCHEDULER    = "cosine"

# ---------------------------------------------------------------------------
# Load base model with 4-bit quantization
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_ID} with 4-bit quantisation...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name       = MODEL_ID,
    max_seq_length   = MAX_SEQ_LEN,
    dtype            = None,       # auto-detect
    load_in_4bit     = True,       # QLoRA
)

# Attach LoRA adapter
model = FastLanguageModel.get_peft_model(
    model,
    r                    = LORA_RANK,
    target_modules       = TARGET_MODULES,
    lora_alpha           = LORA_ALPHA,
    lora_dropout         = LORA_DROPOUT,
    bias                 = "none",
    use_gradient_checkpointing = "unsloth",
    random_state         = 42,
)

total_params  = sum(p.numel() for p in model.parameters())
train_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {train_params:,} / {total_params:,}  ({train_params/total_params*100:.2f}%)")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_sft_dataset(path: Path) -> Dataset:
    """Load finetune_sft.jsonl and format as chat strings."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append(r)

    def to_chat_string(r):
        # Apply the model's chat template
        text = tokenizer.apply_chat_template(
            r["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = Dataset.from_list(records)
    ds = ds.map(to_chat_string)
    return ds


print(f"Loading dataset from {SFT_FILE}...")
dataset = load_sft_dataset(SFT_FILE)
print(f"  {len(dataset)} training examples")

# Optional: filter out very long examples to avoid OOM
def token_len(example):
    return len(tokenizer(example["text"])["input_ids"])

# Split 95/5 train/eval
split = dataset.train_test_split(test_size=0.05, seed=42)
train_ds = split["train"]
eval_ds  = split["test"]
print(f"  Train: {len(train_ds)}  |  Eval: {len(eval_ds)}")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir                  = str(OUTPUT_DIR),
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    learning_rate               = LR,
    lr_scheduler_type           = LR_SCHEDULER,
    warmup_ratio                = WARMUP_RATIO,
    fp16                        = not torch.cuda.is_bf16_supported(),
    bf16                        = torch.cuda.is_bf16_supported(),
    logging_steps               = 10,
    eval_strategy               = "steps",
    eval_steps                  = 100,
    save_strategy               = "steps",
    save_steps                  = 200,
    save_total_limit            = 2,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    report_to                   = "none",
    seed                        = 42,
    dataloader_num_workers      = 2,
    group_by_length             = True,
)

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = train_ds,
    eval_dataset    = eval_ds,
    dataset_text_field = "text",
    max_seq_length  = MAX_SEQ_LEN,
    args            = training_args,
)

print("Starting training...")
trainer.train()

# ---------------------------------------------------------------------------
# Save adapter only (challenge requires ONLY the LoRA adapter)
# ---------------------------------------------------------------------------
adapter_dir = OUTPUT_DIR / "final_adapter"
adapter_dir.mkdir(exist_ok=True)

model.save_pretrained(str(adapter_dir))   # saves adapter_model.safetensors + adapter_config.json
# NOTE: do NOT save tokenizer — submission must contain ONLY adapter files

print(f"\nAdapter saved to {adapter_dir}")
print("Files:")
for f in sorted(adapter_dir.iterdir()):
    size = f.stat().st_size / 1024 / 1024
    print(f"  {f.name:<40} {size:.1f} MB")

# ---------------------------------------------------------------------------
# Package for submission — ONLY adapter_model.safetensors + adapter_config.json
# ---------------------------------------------------------------------------
import zipfile

ALLOWED_FILES = {"adapter_model.safetensors", "adapter_config.json"}

submission_zip = Path("submission.zip")
with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in adapter_dir.iterdir():
        if f.name in ALLOWED_FILES:
            zf.write(f, arcname=f.name)

print(f"\nsubmission.zip created ({submission_zip.stat().st_size/1024/1024:.1f} MB)")
print("Contents:")
with zipfile.ZipFile(submission_zip, "r") as zf:
    for name in zf.namelist():
        print(f"  {name}")
