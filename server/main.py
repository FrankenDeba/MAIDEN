import sys
import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

# Brute-force hardcode the exact directory into Python's memory
hardcoded_root = r"C:\Users\arpan\OneDrive\Documents\MAIDEN"
if hardcoded_root not in sys.path:
    sys.path.insert(0, hardcoded_root)

# Import your newly updated Hybrid Security Firewall
from minja_infer import detect 
from services.forward import forward

app = FastAPI(
    title="MAIDEN Security Gateway",
    description="Dual-endpoint API routing clean traffic to Cloud LLMs or local SLMs after threat filtering."
)

OLLAMA_URL = "http://localhost:11434/api/generate"

class InferenceRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the MAIDEN Hybrid Inference & Security API!"}

@app.post("/infer")
def infer(request: InferenceRequest):
    """
    Cloud Endpoint with Active Hybrid Firewall.
    """
    user_prompt = request.prompt
    
    # 1. Security Check (detect returns True if Malicious)
    is_malicious = detect(user_prompt)
    
    if is_malicious:
        return {
            "input": user_prompt,
            "output": "Echo: {'error': 'Prompt rejected or quarantined due to similarity with known malicious content.'}"
        }
        
    # 2. Forward to Cloud (Groq)
    return {
        "input": user_prompt,
        "output": f"Echo: {forward(user_prompt)}"
    }

@app.post("/infer_slm")
async def infer_slm(request: InferenceRequest):
    """
    Local Edge Endpoint with Active Hybrid Firewall routing to Ollama/Phi-3.
    """
    user_prompt = request.prompt
    
    # 1. Security Check (detect returns True if Malicious)
    is_malicious = detect(user_prompt)
    
    if is_malicious:
        return {
            "input": user_prompt,
            "output": "Echo: {'error': 'Prompt rejected or quarantined due to similarity with known malicious content.'}"
        }
    
    # 2. Forward to Local SLM (Ollama)
    payload = {
        "model": "phi3",
        "prompt": user_prompt,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            slm_response = data.get("response", "No response content generated.")
    except httpx.HTTPStatusError as e:
        slm_response = f"SLM Error: Status code {e.response.status_code}"
    except Exception as e:
        slm_response = f"SLM Connection Error: Ensure Ollama is running. Details: {str(e)}"

    return {
        "input": user_prompt,
        "output": f"SLM_Echo:\n{slm_response}"
    }