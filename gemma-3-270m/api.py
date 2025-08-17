"""
FastAPI service for your LoRA‑tuned Gemma‑3‑270M invoice extractor.

Endpoints
- GET  /health           → simple liveness probe
- GET  /model            → model info
- POST /extract          → body: { invoice_text: str, max_new_tokens?: int }
                           returns { raw_output, parsed_json }

Run locally
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Optional: CORS is enabled for all origins by default. Tighten in production.
"""
from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------
# CONFIG
# -------------------
MODEL_PATH = os.getenv("MODEL_PATH", "./gemma_lora_finetuned")
BASE_MODEL = os.getenv("BASE_MODEL", "google/gemma-3-270m-it")
MAX_NEW_TOKENS_DEFAULT = int(os.getenv("MAX_NEW_TOKENS", "256"))

def get_device_map():
    # Prefer CUDA → MPS → CPU
    if torch.cuda.is_available():
        return "auto"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"": 0}
    return {"": "cpu"}

# -------------------
# MODEL LOAD (once)
# -------------------
print("Loading model...")

device_map = get_device_map()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# Ensure padding token exists (Gemma often uses eos as pad)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map=device_map,
)

model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

# -------------------
# HELPERS
# -------------------
JSON_SCHEMA_KEYS = [
    "invoice_id",
    "date",
    "vendor",
    "customer",
    "total_amount",
    "currency",
]


def extract_first_json(text: str) -> Dict[str, Any]:
    """Extract the first { ... } block; repair quotes/braces; parse to dict.
    Returns {error, raw} if parsing fails.
    """
    match = re.search(r"\{[\s\S]*?\}", text)
    if not match:
        return {"error": "No JSON found", "raw": text}
    snippet = match.group(0)

    # Normalize backticks / code fences if present
    snippet = snippet.strip().strip("`")

    # Replace single quotes with double quotes (best‑effort)
    snippet = snippet.replace("'", '"')

    # Balance braces if model cut off
    open_b = snippet.count("{")
    close_b = snippet.count("}")
    if open_b > close_b:
        snippet += "}" * (open_b - close_b)

    try:
        return json.loads(snippet)
    except Exception as e:
        return {"error": f"Broken JSON snippet ({e})", "raw": snippet}


def normalize_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only expected keys; coerce obvious types; fill missing with None."""
    if "error" in obj:
        return obj
    out: Dict[str, Any] = {}
    for k in JSON_SCHEMA_KEYS:
        v = obj.get(k, None)
        if k == "total_amount" and isinstance(v, str):
            # remove currency symbols/commas
            v = re.sub(r"[^0-9.\-]", "", v)
            try:
                v = float(v) if v != "" else None
            except Exception:
                v = None
        out[k] = v
    return out


def build_prompt(invoice_text: str) -> str:
    return (
        "You are an AI trained to extract structured invoice data.\n\n"
        "Return ONLY valid JSON. Do not include explanations, text, or extra fields.\n\n"
        "Fields to extract:\n"
        "- invoice_id\n- date\n- vendor\n- customer\n- total_amount\n- currency\n\n"
        f"Invoice text:\n{invoice_text}\n\n"
        "JSON:\n"
    )


def generate_response(invoice_text: str, max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT) -> str:
    prompt = build_prompt(invoice_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------
# API
# -------------------
app = FastAPI(title="Gemma‑3 Invoice Extractor", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    invoice_text: str = Field(..., description="Raw invoice text to parse")
    max_new_tokens: Optional[int] = Field(
        default=MAX_NEW_TOKENS_DEFAULT, ge=32, le=1024,
        description="Decoder budget for generation"
    )


class ExtractResponse(BaseModel):
    raw_output: str
    parsed_json: Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_info():
    return {
        "base_model": BASE_MODEL,
        "adapter_path": MODEL_PATH,
        "device": str(model.device),
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
    }


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    raw = generate_response(req.invoice_text, req.max_new_tokens)
    parsed = extract_first_json(raw)
    parsed = normalize_schema(parsed)
    return {"raw_output": raw, "parsed_json": parsed}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
