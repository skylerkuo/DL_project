from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

base_model_path = "Qwen/Qwen2.5-0.5B-Instruct"
lora_checkpoint_path = "outputs/checkpoint-300"
push_dir = "qwen_finetune_v1"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# === Apply LoRA adapter and merge ===
print("Loading LoRA and merging weights...")
model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
model = model.merge_and_unload()

# === Save merged model and tokenizer ===
print(f"Saving merged model to: {push_dir}")
model.save_pretrained(push_dir)
tokenizer.save_pretrained(push_dir)