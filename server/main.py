from fastapi import FastAPI
from pydantic import BaseModel

from minja_infer import detect
app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Inference API!"}

@app.post("/infer")
def infer(request: InferenceRequest):
    # Placeholder for inference logic
    response = {
        "input": request.prompt,
        "output": f"Echo: {detect(request.prompt)}"
    }
    return response