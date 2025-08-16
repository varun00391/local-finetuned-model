# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from transformers import AutoProcessor, AutoModelForImageTextToText
# from PIL import Image
# import torch
# import io

# app = FastAPI(title="Image Description API", description="Upload an image and get a description.")

# # Check device
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")

# # Load model and processor (only once at startup)
# model = AutoModelForImageTextToText.from_pretrained(
#     "./lfm2-450m",
#     torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
#     device_map=device,
#     trust_remote_code=True
# )
# processor = AutoProcessor.from_pretrained("./lfm2-450m", trust_remote_code=True)

# @app.post("/describe")
# async def describe_image(file: UploadFile = File(...)):
#     try:
#         # Load image from upload
#         image_bytes = await file.read()
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#         # Build prompt (with image tokens)
#         prompt = "<|image_start|><image><|image_end|> Describe the scene in the image, including the setting, objects, and any activities."

#         # Apply chat template if available
#         if hasattr(processor, "apply_chat_template"):
#             messages = [{"role": "user", "content": prompt}]
#             prompt = processor.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)

#         # Preprocess inputs
#         inputs = processor(
#             text=prompt,
#             images=[image],
#             return_tensors="pt"
#         ).to(device)

#         # Generate description
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=200,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             eos_token_id=processor.tokenizer.eos_token_id
#         )

#         description = processor.batch_decode(outputs, skip_special_tokens=True)[0]

#         return JSONResponse({"description": description})

#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import io

app = FastAPI(title="Image Description API", description="Upload an image and get a description.")

# Check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and processor (only once at startup)
model = AutoModelForImageTextToText.from_pretrained(
    "./lfm2-450m",
    torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
    device_map=device,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("./lfm2-450m", trust_remote_code=True)

@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    try:
        # Load image from upload
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Build prompt (with image tokens)
        prompt = "<|image_start|><image><|image_end|> Describe the scene in the image, including the setting, objects, and any activities."

        # Apply chat template if available
        if hasattr(processor, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # Added add_generation_prompt=True

        # Preprocess inputs
        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        ).to(device)

        # Generate description
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=processor.tokenizer.eos_token_id
        )

        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # --- NEW CODE ADDED HERE ---
        # The prompt is part of the output string. We need to remove it.
        # Find the end of the prompt string and slice the generated text.
        # The chat template adds "user\n" before the prompt, so we need to account for that.
        prompt_end_marker = "Describe the scene in the image, including the setting, objects, and any activities."
        if prompt_end_marker in generated_text:
            description_start_index = generated_text.find(prompt_end_marker) + len(prompt_end_marker)
            description = generated_text[description_start_index:].strip()
        else:
            # Fallback in case the prompt isn't found
            description = generated_text.strip()
            
        # --- END OF NEW CODE ---

        return JSONResponse({"description": description})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)