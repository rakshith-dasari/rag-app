import requests
import os
from dotenv import load_dotenv
# Ensure you have the necessary environment variables set
load_dotenv()

def ask_llm_with_context(query: str, context_chunks: list[str]) -> str:
    system_prompt = (
        "You are a helpful assistant answering questions based strictly on the provided context. "
        "The context is from insurance policy documents. Do not use outside knowledge."
    )

    user_prompt = f"""Answer the following question using only the provided context.

Question: {query}

Context:
{chr(10).join(context_chunks)}

Answer:"""

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek/deepseek-r1-distill-llama-70b:free",  # You can parameterize this
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM request failed: {e}"
