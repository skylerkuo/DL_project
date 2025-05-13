# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import transformers
import pandas as pd
from collections import Counter

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

raw_ds = load_dataset("sentiment140")
raw_ds = raw_ds.filter(lambda x: x["sentiment"] in [0, 4])

print(raw_ds["train"].column_names)
for i in range(3):
    print(raw_ds["train"][i])

print(Counter(raw_ds["train"]["sentiment"]))


def formatting_func(example):
    return f"""You are a helpful assistant. you need to tell me the following sentence is positive or negative, only reply with positive or negative.sentence:{example['text']}. """

def tokenize_fn(example):
    prompt = formatting_func(example)
    inputs = tokenizer(prompt, truncation=True, padding='max_length', max_length=200)
    labels = tokenizer("positive" if example['sentiment'] == 4 else "negative", truncation=True, padding='max_length', max_length=80)
    inputs['labels'] = labels['input_ids']
    return inputs

train_data = raw_ds['train'].select(range(10000)).map(tokenize_fn, batched=False)
test_data  = raw_ds['test'].map(tokenize_fn, batched=False)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
import bitsandbytes as bnb

def find_all_linear_names(m):
    cls = bnb.nn.Linear4bit
    names = set()
    for n, module in m.named_modules():
        if isinstance(module, cls):
            parts = n.split('.')
            names.add(parts[-1])
    if 'lm_head' in names:
        names.remove('lm_head')
    return list(names)

lora_targets = find_all_linear_names(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=lora_targets,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    formatting_func=formatting_func,
    peft_config=lora_config,
    args=transformers.TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=3,
        max_steps=200,
        learning_rate=1e-4,
        logging_steps=10,
        optim="paged_adamw_8bit",
        report_to="none",
        save_strategy="epoch"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False
trainer.train()