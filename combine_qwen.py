from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

base_model_path = "Qwen/Qwen2.5-0.5B-Instruct"
lora_checkpoint_path = "outputs/checkpoint-6000"
push_dir = "qwen_finetune_v1"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_path,
    num_labels=2,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("Loading LoRA and merging weights...")
model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
model = model.merge_and_unload()

print(f"Saving merged model to: {push_dir}")
model.save_pretrained(push_dir)
tokenizer.save_pretrained(push_dir)
