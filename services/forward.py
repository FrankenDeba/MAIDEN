from .interceptor import intercept
from .llm import ask_llm

def forward(prompt):
    if intercept(prompt)["predicted_label"] not in ["malicious", "high_risk"]:
        print("[!] Prompt flagged as benign, forwarding to LLM...")
        return ask_llm(prompt)
    else:
        return {
            "error": "Prompt rejected or quarantined due to similarity with known malicious content."
        }