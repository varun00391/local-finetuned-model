import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# -------------------
# CONFIG
# -------------------
MODEL_NAME = "google/gemma-3-270m-it"
# DATASET_NAME = "Abirate/english_quotes"  # Example dataset
OUTPUT_DIR = "./gemma_lora_finetuned"

# LoRA settings
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Layers to apply LoRA on
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# -------------------
# LOAD MODEL & TOKENIZER
# -------------------
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"  # Use eager attention as recommended
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# -------------------
# DATA PREPARATION
# -------------------
print("Loading dataset...")
# dataset = load_dataset(DATASET_NAME)
dataset = load_dataset("json", data_files="data/invoice_dataset.json")

# Split the 'train' dataset into train and test (90% train, 10% test)
dataset = dataset["train"].train_test_split(test_size=0.1)

# Tokenization function
# def tokenize_function(examples):
#     tokenized = tokenizer(
#         examples["quote"],
#         padding="max_length",
#         truncation=True,
#         max_length=128
#     )
#     # Add labels as a copy of input_ids for causal language modeling
#     tokenized["labels"] = tokenized["input_ids"].copy()
#     return tokenized

def tokenize_function(examples):
    # Format the text as an instruction-response pair
    texts = [
        f"Instruction: {instr}\nInput: {inp}\nResponse: {out}"
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=256
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(
    [col for col in tokenized_datasets["train"].column_names if col not in ["input_ids", "attention_mask", "labels"]]
)
tokenized_datasets.set_format("torch")

# -------------------
# TRAINING
# -------------------
print("Starting training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    fp16=False,
    bf16=False,
    save_total_limit=2,
    push_to_hub=False,
    dataloader_pin_memory=False  # Disable pin_memory for MPS compatibility
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    # processing_class=tokenizer  # Updated to avoid deprecation warning
    tokenizer=tokenizer
)

trainer.train()

# -------------------
# SAVE MODEL
# -------------------
print(f"Saving LoRA fine-tuned model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete ✅")