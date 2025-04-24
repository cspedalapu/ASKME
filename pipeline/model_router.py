# Model Router for AI Assistant
# This module routes queries to different AI models based on the model name.
# It supports OpenAI's GPT-4, Ollama's LLaMA3, and a placeholder for DeepSeek.

import os
import openai
import requests
from openai import OpenAI

# Set keys only once
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_answer(model_name: str, context: str, query: str) -> str:
    if model_name == "openai-gpt4":
        return ask_openai_gpt4(context, query)
    elif model_name == "ollama-llama3":
        return ask_ollama_llama3(context, query)
    elif model_name == "deepseek":
        return ask_deepseek_gpt(context, query)
    else:
        return "Model not supported yet."

# Model Implementations
client = OpenAI()
def ask_openai_gpt4(context, query):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use only the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ OpenAI GPT-4 error: {e}"
    

def ask_ollama_llama3(context, query):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": f"Context:\n{context}\n\nQuestion: {query}",
                "stream": False
            }
        )
        return response.json().get("response", "⚠️ No response from Ollama.")
    except Exception as e:
        return f"Ollama LLaMA3 error: {e}"

def ask_deepseek_gpt(context, query):
    # Placeholder for future DeepSeek API integration
    return "(DeepSeek is coming soon... 🚧)"
