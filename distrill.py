# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch, math, evaluate, os
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    logging
)
import matplotlib.pyplot as plt

logging.set_verbosity_warning()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEACHER_MODEL_PATH = "./qwen_lora_seqcls/final"
STUDENT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "./student_qwen0.5b_distilled"
ALPHA = 0
TEMPERATURE = 2.0
NUM_TRAIN = 50000

label2id = {"negative": 0, "positive": 1}
id2label = {0: "negative", 1: "positive"}

raw_ds = load_dataset("sentiment140")
raw_ds = raw_ds.filter(lambda x: x["sentiment"] in [0, 4])  # binary

tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
train_ds = tokenized_ds["train"].shuffle(seed=42).select(range(NUM_TRAIN))
eval_ds  = tokenized_ds["test"].shuffle(seed=42)
collator = DataCollatorWithPadding(tokenizer)

teacher = AutoModelForSequenceClassification.from_pretrained(
    TEACHER_MODEL_PATH,
    trust_remote_code=True
).to(device)
teacher.eval()
student = AutoModelForSequenceClassification.from_pretrained(
    STUDENT_MODEL_ID,
    num_labels=2,
    trust_remote_code=True
).to(device)

class DistillTrainer(Trainer):
    def __init__(self, teacher_model: nn.Module, alpha: float, T: float, **kw):
        super().__init__(**kw)
        self.teacher = teacher_model
        self.alpha = alpha
        self.T = T
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        with torch.no_grad():
            t_logits = self.teacher(**inputs).logits

        outputs = model(**inputs)
        s_logits = outputs.logits
        ce_loss = nn.functional.cross_entropy(s_logits, labels)

        # 蒸餾 loss
        t_logits /= self.T
        s_logits /= self.T
        kl_loss = self.kl(
            nn.functional.log_softmax(s_logits, dim=-1),
            nn.functional.softmax(t_logits, dim=-1)
        ) * (self.T ** 2)

        loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        return (loss, outputs) if return_outputs else loss

metric_acc = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return metric_acc.compute(predictions=preds, references=labels)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-5,
    eval_steps=100,
    save_steps=100,
    max_steps=10000,
    fp16=True,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to="none"
)

trainer = DistillTrainer(
    model=student,
    teacher_model=teacher,
    alpha=ALPHA,
    T=TEMPERATURE,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

log_history = trainer.state.log_history
train_loss, eval_loss, eval_acc, epochs = [], [], [], []

for log in log_history:
    if 'loss' in log and 'epoch' in log:
        train_loss.append(log['loss'])
        epochs.append(log['epoch'])
    if 'eval_loss' in log:
        eval_loss.append(log['eval_loss'])
    if 'eval_accuracy' in log:
        eval_acc.append(log['eval_accuracy'])

plt.figure()
plt.plot(epochs[:len(train_loss)], train_loss, label='Train Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png")

plt.figure()
plt.plot(epochs[:len(eval_acc)], eval_acc, label='Eval Accuracy', color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Evaluation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/accuracy_curve.png")

trainer.save_model(f"{OUTPUT_DIR}/final_student_onlyce")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_student_onlyce")
print(f"模型儲存：{OUTPUT_DIR}")
