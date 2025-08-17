import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------
# CONFIG
# -------------------
MODEL_PATH = "./gemma_lora_finetuned"  # path where your LoRA model is saved

# -------------------
# LOAD MODEL & TOKENIZER
# -------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-270m-it",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# load LoRA weights
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

# -------------------
# HELPERS
# -------------------
def extract_first_json(text: str):
    """
    Extract the first JSON object { ... } from model output.
    Fix quotes and unbalanced braces if needed.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"error": "No JSON found", "raw": text}

    snippet = match.group(0)

    # replace single quotes → double quotes
    snippet = snippet.replace("'", '"')

    # balance braces if model cut off
    open_braces = snippet.count("{")
    close_braces = snippet.count("}")
    if open_braces > close_braces:
        snippet += "}" * (open_braces - close_braces)

    try:
        return json.loads(snippet)
    except Exception as e:
        return {"error": f"Broken JSON snippet ({e})", "raw": snippet}


def generate_response(invoice_text, max_new_tokens=256):
    """Generate structured JSON output from invoice text."""

    # Much cleaner instruction
    prompt = f"""
You are an AI trained to extract structured invoice data.

Return ONLY valid JSON. Do not include explanations, text, or extra fields.

Fields to extract:
- invoice_id
- date
- vendor
- customer
- total_amount
- currency

Invoice text:
{invoice_text}

JSON:
"""

    inputs = tokenizer(prompt.strip(), return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,              # deterministic
            eos_token_id=tokenizer.eos_token_id  # stop at EOS
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------
# TEST CASE
# -------------------
if __name__ == "__main__":
    sample_invoice = """Invoice ID: 5
Date: 2025-02-01
Vendor: Blue Corp
Address: 123 Main St, NY
Customer: John Smith
Customer Address: 88 Park Ave, NY
Items:
  1. Laptop - 2 pcs - $1200.50
  2. Mouse - 5 pcs - $50.00
Total: $1250.50"""

    raw_output = generate_response(sample_invoice)

    # print("\n=== Raw Model Output ===")
    # print(raw_output)

    parsed = extract_first_json(raw_output)
    print("\n=== Parsed JSON ===")
    print(json.dumps(parsed, indent=2))
