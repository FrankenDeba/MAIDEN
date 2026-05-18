import httpx

# URL where your local Ollama SLM instance runs
OLLAMA_URL = "http://localhost:11434/api/generate"

async def query_slm(prompt: str, model_name: str = "phi3") -> str:
    """
    Sends a safe prompt to the local Small Language Model (SLM) via Ollama.
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False  # Set to False to get the whole response at once
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "No response content generated.")
            
    except httpx.HTTPStatusError as e:
        return f"SLM Error: Failed to communicate with local model. Status code: {e.response.status_code}"
    except Exception as e:
        return f"SLM Connection Error: Is Ollama running? Details: {str(e)}"