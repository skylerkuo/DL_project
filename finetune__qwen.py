# -*- coding: utf-8 -*-
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate

model_id = "Qwen/Qwen2.5-1.5B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "dense"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(model, lora_config)

raw_ds = load_dataset("sentiment140")
raw_ds = raw_ds.filter(lambda x: x["sentiment"] in [0, 4])

def preprocess(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokenized["label"] = 1 if example["sentiment"] == 4 else 0
    return tokenized

tokenized_ds = raw_ds.map(preprocess, remove_columns=["date", "query", "user", "text", "sentiment"])
train_ds = tokenized_ds["train"].shuffle(seed=42).select(range(50000))
eval_ds  = tokenized_ds["test"].shuffle(seed=42)
acc_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    return acc_metric.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="qwen_lora_seqcls",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    logging_steps=10,
    eval_steps=200,
    save_steps=200,
    max_steps=20000,
    logging_dir="logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("qwen_lora/final")
tokenizer.save_pretrained("qwen_lora/final")
