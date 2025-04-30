#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch, math, evaluate
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    logging,
)

logging.set_verbosity_warning()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEACHER_NAME = "./teacher_sent140/checkpoint-30000"  #可以選一個差不多的checkpoint
NUM_STU_LAYERS = 4
ALPHA = 0.5
TEMPERATURE = 2.0
OUTPUT_DIR = "./student_sent140"


raw_ds = load_dataset("sentiment140")
raw_ds = raw_ds.filter(lambda x: x["sentiment"] in [0, 4])
tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)

def preprocess(ex): return tokenizer(ex["text"], truncation=True)
tokenized = raw_ds.map(
    preprocess, batched=True,
    remove_columns=["date", "query", "user", "text"]
)

def remap_label(examples):
    mapping = {0: 0, 4: 1}
    return {"label": mapping[examples["sentiment"]]}
tokenized = tokenized.map(remap_label, remove_columns=["sentiment"])

train_ds = tokenized["train"].shuffle(seed=42).select(range(10000)) 
eval_ds  = tokenized["test"].shuffle(seed=42).select(range(300))

collator = DataCollatorWithPadding(tokenizer)

teacher = AutoModelForSequenceClassification.from_pretrained(
    TEACHER_NAME, num_labels=2
).to(device)
teacher.eval()

def build_student_from_teacher(teacher_model, num_layers:int):
    teacher_cfg: BertConfig = teacher_model.config
    stu_cfg = BertConfig.from_dict(teacher_cfg.to_dict())
    stu_cfg.num_hidden_layers = num_layers
    stu_cfg.num_labels = 2                
    student = AutoModelForSequenceClassification.from_config(stu_cfg)

    student.bert.embeddings.load_state_dict(teacher_model.bert.embeddings.state_dict())
    student.bert.pooler.load_state_dict(teacher_model.bert.pooler.state_dict())
    student.classifier.load_state_dict(teacher_model.classifier.state_dict())

    sel = torch.linspace(0, teacher_cfg.num_hidden_layers-1, num_layers).long()
    for i, idx in enumerate(sel):
        student.bert.encoder.layer[i].load_state_dict(
            teacher_model.bert.encoder.layer[idx.item()].state_dict()
        )
    return student

student = build_student_from_teacher(teacher, NUM_STU_LAYERS).to(device)

class DistillTrainer(Trainer):
    def __init__(self, teacher_model: nn.Module, alpha: float, T: float, **kw):
        super().__init__(**kw)
        self.teacher = teacher_model
        self.alpha   = alpha
        self.T       = T
        self.kl      = nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        with torch.no_grad():
            t_logits = self.teacher(**inputs).logits

        outputs = model(**inputs)
        s_logits = outputs.logits
        ce_loss = nn.functional.cross_entropy(s_logits, labels)

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
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-6,
    num_train_epochs=40,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    fp16=False,
)

trainer = DistillTrainer(
    model=student,
    teacher_model=teacher,
    alpha=ALPHA,
    T=2,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final_student")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_student")
print("✅ 三分類蒸餾完成，模型已存至：", f"{OUTPUT_DIR}/final_student")
