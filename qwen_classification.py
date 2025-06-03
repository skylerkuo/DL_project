import torch
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "Qwen/Qwen2.5-1.5B" #模型路徑自己換
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map="cuda").eval()

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
test_ds = tokenized_ds["train"].shuffle(seed=32).select(range(10000))

correct = 0
start_time = time.time()

for i in range(len(test_ds)):
    input_ids = torch.tensor(test_ds[i]["input_ids"]).unsqueeze(0).to(model.device)
    attention_mask = torch.tensor(test_ds[i]["attention_mask"]).unsqueeze(0).to(model.device)
    label = test_ds[i]["label"]

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred = torch.argmax(logits, dim=-1).item()

    if pred == label:
        correct += 1

    print(f"[{i+1}/{len(test_ds)}]  GT: {label}  Pred: {pred}")

end_time = time.time()
total_time = end_time - start_time
avg_time = total_time / len(test_ds)

print(f"Accuracy : {correct / len(test_ds):.2%}")
print(f"Total inference time: {total_time:.2f} seconds")
print(f"Avg. time per sample: {avg_time * 1000:.2f} ms")
