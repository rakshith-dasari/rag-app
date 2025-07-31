import requests
import os
from dotenv import load_dotenv
import logging
from openai import OpenAI
from google import genai


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

    user_prompt = f"""Answer the following question using only the provided context. Don't make up any information or provide any additional context outside of the provided text.  The output should be consistent, interpretable, and usable for downstream applications such as claim processing or audit tracking.
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
        ## Open Router
        # response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
        # response.raise_for_status()
        # llm_response = response.json()["choices"][0]["message"]["content"]

        ## OpenAI
        # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response = client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt}
        #     ]
        # )
        # llm_response = response.choices[0].message.content.strip()

        ## Google GenAI
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt
        )
        llm_response = response.text.strip()
        logging.info("Received response from LLM.")
        return llm_response
    except Exception as e:
        logging.error(f"LLM request failed: {e}", exc_info=True)
        return f"LLM request failed: {e}"
