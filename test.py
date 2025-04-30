# 測試任何模型(教師、學生)都可以用這個來測試

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch, math, evaluate
from torch import nn
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    logging,
)

logging.set_verbosity_warning()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "./student_sent140/checkpoint-12500" #要測試哪個模型就用哪個

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

test_num = 1000

data = load_dataset("sentiment140")
data = data.shuffle(seed=2)

def predict(text):
    input = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**input)
        logits = output.logits
        prediction = torch.argmax(logits, dim=-1)
    return prediction

correct = 0

for i in range(test_num):
    ans = predict(data['train'][i]['text'])
    true_label = 0 if data['train'][i]['sentiment'] == 0 else 1
    if ans[0].item() == true_label:
        correct = correct + 1

accuracy = correct/test_num
print(accuracy)
    


    



