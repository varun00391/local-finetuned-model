from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Path to the downloaded model folder
model_path = "./gemma-3-270m-it"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model in float32 for CPU
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # Full precision to avoid NaNs
    device_map="cpu"            # Force CPU usage
)

# Test prompt
# prompt = "Write a short poem about AI and nature."
prompt = "You are a travel guide. Recommend 3 places to visit in Singapore and explain why."

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
outputs = model.generate(
    **inputs,
    max_length=500,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# Decode and print
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
