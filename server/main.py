import sys
import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from services.slm import query_slm

# Import your newly updated Hybrid Security Firewall
from minja_infer import detect 
from services.forward import forward

app = FastAPI(
    title="MAIDEN Security Gateway",
    description="Dual-endpoint API routing clean traffic to Cloud LLMs or local SLMs after threat filtering."
)

from fastapi.middleware.cors import CORSMiddleware

# Define the origins that are allowed to make requests to your API
origins = [
    "http://localhost:3000",  # Common React port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # List of allowed origins
    allow_credentials=True,           # Allow cookies and auth headers
    allow_methods=["*"],              # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],              # Allow all headers
)

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
    is_malicious = detect(user_prompt)["clasasification"] == "malicious"
    
    if is_malicious:
        return {
            "input": user_prompt,
            "output": "Echo: {'error': 'Prompt rejected or quarantined due to similarity with known malicious content.'}"
        }
    
    try:
        slm_response = query_slm(user_prompt)
    except httpx.HTTPStatusError as e:
        slm_response = f"SLM Error: Status code {e.response.status_code}"
    except Exception as e:
        slm_response = f"SLM Connection Error: Ensure Ollama is running. Details: {str(e)}"

    return {
        "input": user_prompt,
        "output": f"SLM_Echo:\n{slm_response}"
    }