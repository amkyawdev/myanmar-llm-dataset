#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Qwen/Qwen2-0.5B-Instruct (Fixed Version)
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer

# --- CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
OUTPUT_DIR = "./lora_myanmar_chat"
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
# warmup_ratio အစား warmup_steps ကို သုံးခြင်းက ပိုတည်ငြိမ်ပါတယ်
WARMUP_STEPS = 10 

print("🚀 Starting LoRA Fine-tuning for Qwen/Qwen2-0.5B-Instruct...")

# Check GPU
if not torch.cuda.is_available():
    print("❌ ERROR: No GPU detected!")
    exit(1)

print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")

# --- LOAD MODEL AND TOKENIZER ---
print("\n📥 Loading Model and Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # torch_dtype နေရာမှာ dtype လို့ ပြောင်းသုံးပါ (Warning မတက်အောင်)
    dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ Model loaded successfully!")

# --- LOAD DATASET ---
print("\n📂 Loading Dataset from local files...")
# local file တွေရှိမရှိ အရင်စစ်ဆေးပါ
dataset = load_dataset("json", data_files={
    "train": f"{DATASET_PATH}/train.jsonl",
    "validation": f"{DATASET_PATH}/validation.jsonl",
    "test": f"{DATASET_PATH}/test.jsonl"
})

# --- CONFIGURE LORA ---
print("\n⚙️ Configuring LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- TRAINING ARGUMENTS ---
print("\n🎯 Setting up Training Arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS, # Ratio အစား Steps သုံးထားသည်
    logging_steps=10,
    save_steps=100,
    eval_strategy="epoch", # eval_strategy အစား evaluation_strategy ကို သုံးနိုင်သည်
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
)

# --- FORMATTING DATASET ---
def format_messages(example):
    """Convert messages format to text for SFT"""
    text = ""
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return {"text": text}

dataset = dataset.map(format_messages)

# --- TRAINER ---
print("\n🏋️ Starting Training...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    # CRITICAL FIX: max_length အစား max_seq_length ကို သုံးပါ
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text", # text column ကို အသုံးပြုရန် သတ်မှတ်ပေးရပါမည်
)

# Train
trainer.train()

# --- SAVE MODEL ---
print("\n💾 Saving Model...")
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print(f"\n✅ Training Complete!")
