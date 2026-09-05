import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Prannesshkva/OrthoSSM-130M"
print(f"Loading {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32)

prompt = "Orthogonal state-space models enable"
inputs = tokenizer(prompt, return_tensors="pt")

print("Generating tokens with exact isometric dynamics...")
outputs = model.generate(**inputs, max_new_tokens=40, do_sample=False)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Output:")
print(result)
