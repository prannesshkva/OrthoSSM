import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_quphantom_demo():
    print("=" * 70)
    print("🦅 QuPhantom (QU-PHANTOM) Architecture Quickstart")
    print("Author: Prannessh K.V.A. (@Prannesshkva)")
    print("DOI: 10.5281/zenodo.22177116")
    print("=" * 70)

    model_id = "Prannesshkva/QuPhantom-Mamba-Falcon-Hybrid"
    print(f"Loading {model_id} from Hugging Face Hub...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()

    prompt = "Explain why Quasi-Unitarity ensures numerical stability in continuous-time state-space models:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"\nPrompt: \"{prompt}\"")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )

    gen_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    print(f"\nGenerated Response:\n{response}")

if __name__ == "__main__":
    run_quphantom_demo()
