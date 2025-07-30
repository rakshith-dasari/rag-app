import requests
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("generation.log"),
                        logging.StreamHandler()
                    ])

# Ensure you have the necessary environment variables set
load_dotenv()

def ask_llm_with_context(query: str, context_chunks: list[str]) -> str:
    logging.info(f"Asking LLM with query: '{query}'")
    system_prompt = (
        "Answer using only the provided insurance‑policy context. "
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
        "model": os.getenv("LLM_MODEL"),  # You can parameterize this
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        logging.info("Sending request to LLM...")
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        llm_response = response.json()["choices"][0]["message"]["content"]
        logging.info("Received response from LLM.")
        return llm_response
    except Exception as e:
        logging.error(f"LLM request failed: {e}", exc_info=True)
        return f"LLM request failed: {e}"
