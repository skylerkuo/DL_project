import json
from datasets import Dataset
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    device_map="auto",
).eval()

model = torch.compile(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def llm_reply(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. you need to tell me the following is positive or negative, only reply with positive or negative."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id
        )

    output_ids = generated_ids[0][model_inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return response

ans = llm_reply("you are a good person!")
print(ans)