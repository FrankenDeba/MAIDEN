import os
from groq import Groq

# 🚨 HACKATHON HOTFIX: 
# It will try to find the environment variable first. 
# If it fails, it will use the hardcoded string you paste below.
API_KEY = os.environ.get("GROQ_API_KEY", "gsk_YOUR_ACTUAL_API_KEY_HERE")

client = Groq(api_key=API_KEY)

def forward(prompt: str) -> str:
    """
    Forwards a clean, pre-filtered prompt to the Groq LLM.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant", # Or your specific Groq model
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Cloud LLM API Error: {str(e)}"