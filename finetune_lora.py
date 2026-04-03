#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for ShweYon-V3-Base with Myanmar Chat Dataset

⚠️ REQUIREMENTS:
- GPU with at least 16GB VRAM
- Python packages: torch, transformers, peft, trl, accelerate, datasets

INSTALL:
pip install torch transformers peft trl accelerate datasets

RUN ON GOOGLE COLAB:
https://colab.research.google.com/
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer

# --- CONFIGURATION ---
# Mistral 7B - powerful chat model for Myanmar
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "./lora_myanmar_chat"
# Load dataset from local files (GitHub repo cloned)
DATASET_PATH = "./data/processed"

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Training Configuration
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512
WARMUP_RATIO = 0.1

print("🚀 Starting LoRA Fine-tuning for ShweYon-V3-Base...")
print(f"📁 Model: {MODEL_NAME}")
print(f"📁 Dataset: {DATASET_PATH}")

# Check GPU
if not torch.cuda.is_available():
    print("❌ ERROR: No GPU detected!")
    print("   Please run this on a GPU machine or Google Colab")
    exit(1)

print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# --- LOAD MODEL AND TOKENIZER ---
print("\n📥 Loading Model and Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Use FP16 for better memory efficiency
    device_map="auto",
    trust_remote_code=True
)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ Model loaded successfully!")

# --- LOAD DATASET ---
print("\n📂 Loading Dataset from local files...")
dataset = load_dataset("json", data_files={
    "train": f"{DATASET_PATH}/train.jsonl",
    "validation": f"{DATASET_PATH}/validation.jsonl",
    "test": f"{DATASET_PATH}/test.jsonl"
})

print(f"✅ Train samples: {len(dataset['train'])}")
print(f"✅ Validation samples: {len(dataset['validation'])}")
print(f"✅ Test samples: {len(dataset['test'])}")

# --- CONFIGURE LORA ---
print("\n⚙️ Configuring LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("✅ LoRA configured successfully!")

# --- TRAINING ARGUMENTS ---
print("\n🎯 Setting up Training Arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=10,
    save_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
)

# --- TRAINER ---
print("\n🏋️ Starting Training...")

# Format dataset for SFT - convert messages to text
def format_messages(example):
    """Convert messages format to text for SFT"""
    text = ""
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"System: {content}\n"
        elif role == "user":
            text += f"User: {content}\n"
        elif role == "assistant":
            text += f"Assistant: {content}<|endoftext|>\n"
    return {"text": text}

# Apply formatting
dataset = dataset.map(format_messages, remove_columns=["messages"])

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
)

# Train
trainer.train()

# --- SAVE MODEL ---
print("\n💾 Saving Model...")
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print(f"\n✅ Training Complete!")
print(f"📁 Model saved to: {OUTPUT_DIR}/final")
print(f"\n📝 To use the fine-tuned model:")
print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
print(f"   model = AutoModelForCausalLM.from_pretrained('{OUTPUT_DIR}/final')")
print(f"   tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}/final')")