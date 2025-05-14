import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from datasets import load_dataset

model_name = "qwen_finetune_v1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
).eval()

raw_ds = load_dataset("sentiment140")
raw_ds = raw_ds.filter(lambda x: x["sentiment"] in [0, 4]).shuffle(seed=42)
test_ds = raw_ds["train"].select(range(200))

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    return "positive" if pred == 1 else "negative", pred

correct = 0
for i in range(len(test_ds)):
    prompt = test_ds[i]["text"]
    label = "positive" if test_ds[i]["sentiment"] == 4 else "negative"

    pred_str, pred_id = classify_sentiment(prompt)

    print(f"TEXT: {prompt}")
    print(f"ANSWER: {label} | LLM: {pred_str}")
    print("-" * 40)

    if pred_str == label:
        correct += 1

print(f"Accuracy: {correct / len(test_ds):.2%}")
