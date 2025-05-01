# -*- coding: utf-8 -*-

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    logging,
)
import evaluate
from collections import Counter

logging.set_verbosity_warning()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEACHER_MODEL_NAME = "bert-base-uncased"
TEACHER_OUTPUT_DIR = "./teacher_sent140"

raw_ds = load_dataset("sentiment140")
raw_ds = raw_ds.filter(lambda x: x["sentiment"] in [0, 4])

print(raw_ds["train"].column_names)
for i in range(3):
    print(raw_ds["train"][i])

print(Counter(raw_ds["train"]["sentiment"]))

tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized = raw_ds.map(
    preprocess, batched=True,
    remove_columns=["date", "query", "user", "text"]
)

def remap_label(examples):
    mapping = {0: 0, 4: 1}
    return {"label": mapping[examples["sentiment"]]}

tokenized = tokenized.map(remap_label, remove_columns=["sentiment"])

print("after mapping training dataset label：", Counter(tokenized["train"]["label"]))
print("after mapping testing dataset label：", Counter(tokenized["test"]["label"]))

train_ds = tokenized["train"].shuffle(seed=42).select(range(10000))
eval_ds = tokenized["test"].shuffle(seed=42)

collator = DataCollatorWithPadding(tokenizer)

teacher_model = AutoModelForSequenceClassification.from_pretrained(
    TEACHER_MODEL_NAME, num_labels=2
).to(device)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return metric.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir=TEACHER_OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-6,
    num_train_epochs=40,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    fp16=False,
)

trainer = Trainer(
    model=teacher_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(f"{TEACHER_OUTPUT_DIR}/final_teacher")
tokenizer.save_pretrained(f"{TEACHER_OUTPUT_DIR}/final_teacher")

print("訓練完成，存到：", f"{TEACHER_OUTPUT_DIR}/final_teacher")
