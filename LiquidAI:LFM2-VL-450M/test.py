from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image

# Check device (MPS if available, else CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
try:
    model = AutoModelForImageTextToText.from_pretrained(
        "./lfm2-450m",
        torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained("./lfm2-450m", trust_remote_code=True)
except Exception as e:
    print(f"Error loading model or processor: {e}")
    exit(1)

# Load image
try:
    image = Image.open("1000058611.jpg").convert("RGB")  #1000058611.jpg, 1*G9wRBBlFCEhABSAH2nodIg.png
except FileNotFoundError:
    print("Error: Image file not found.")
    exit(1)

# Prepare inputs with a conversational prompt
# prompt = "<|image_start|><image><|image_end|> Describe this image in detail."  # Use image start/end tokens
# prompt = "<|image_start|><image><|image_end|> Describe the image in at least 50 words, covering objects, colors, background, and any unique features."
# prompt = "List all elements in the image and describe their appearance in detail."
prompt = "<|image_start|><image><|image_end|> Describe the scene in the image, including the setting, objects, and any activities."
try:
    # Apply chat template if available
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {"role": "user", "content": f"<|image_start|><image><|image_end|> Describe the scene in the image, including the setting, objects, and any activities."}
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False)
        # print("Applied chat template:", prompt)
    
    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt"
    )
    inputs = inputs.to(device)
except Exception as e:
    print(f"Error processing inputs: {e}")
    exit(1)

# Generate output with adjusted parameters
try:
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,  # Enable sampling for more diverse output
        temperature=0.7,  # Control randomness
        top_p=0.9,  # Use nucleus sampling
        eos_token_id=processor.tokenizer.eos_token_id  # Ensure proper termination
    )
    print("Generated output:", processor.batch_decode(outputs, skip_special_tokens=True)[0])
except Exception as e:
    print(f"Error during generation: {e}")
    exit(1)