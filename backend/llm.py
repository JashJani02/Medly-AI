import requests
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = "gemma4:e4b" #* Can change models from here. 


def generate_response(prompt):

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 2048
                }
            },
            timeout=120
        )

        data = response.json()

        return data.get("response", "Sorry, I couldn't generate a response.")

    except Exception as e:
        print("LLM ERROR:", e)
        return "Sorry, the AI model is not responding right now."
