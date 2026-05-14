import os 
import dotenv

dotenv.load_dotenv()

def ask_llm(prompt):
    from groq import Groq

    print("[!] Sending prompt to LLM...")
    print(f"Prompt: {prompt}")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
        {
            "role": "user",
            "content": prompt
        }
        ],
        temperature=1,
        max_completion_tokens=4032,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None
    )

    answer = ""

    for chunk in completion:
        answer += chunk.choices[0].delta.content or ""

    return answer

__all__ = ["ask_llm"]
