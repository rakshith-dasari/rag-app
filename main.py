from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from generation import ask_llm_with_context
from faiss_kb import search_faiss_index, generate_kb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("main.log"),
                        logging.StreamHandler()
                    ])

load_dotenv()

app = FastAPI()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pdf-index"
NAMESPACE = "__default__"

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(INDEX_NAME)

class SearchRequest(BaseModel):
    query: str
    pdf_url: str
    top_k: int = 5

@app.post("/search")
def run_search(request: SearchRequest):
    try:
        logging.info(f"Received search request with query: {request.query}")
        query_dict = {
            "inputs": {"text": request.query},
            "top_k": request.top_k
        }

        result = pinecone_index.search(
            namespace=NAMESPACE,
            query=query_dict,
            fields=["chunk_text"]
        )
        context_chunks = [
            hit["fields"].get("chunk_text", "")
            for hit in result["result"]["hits"]
        ]
        logging.info(f"Found {len(context_chunks)} chunks from pinecone index.")

        response = ask_llm_with_context(query=request.query, context_chunks=context_chunks)
        logging.info("Returning response from LLM.")
        return response

    except Exception as e:
        logging.error(f"Error during search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/faiss")
def run_faiss_search(request: SearchRequest):
    try:
        logging.info(f"Received faiss search request with query: {request.query}")
        context_chunks = search_faiss_index(request.query, request.pdf_url, request.top_k)
        logging.info(f"Found {len(context_chunks)} chunks from faiss index.")
        response = ask_llm_with_context(query=request.query, context_chunks=context_chunks)
        logging.info("Returning response from LLM.")
        return response

    except Exception as e:
        logging.error(f"Error during faiss search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class HackRxRequest(BaseModel):
    documents: str
    questions: list[str]

@app.post("/api/v1/hackrx/run")
def run_hackrx(request: HackRxRequest):
    try:
        logging.info(f"Received hackrx request with {len(request.questions)} questions.")
        generate_kb(request.documents)
        logging.info(f"Knowledge base generated for {request.documents}")
        final_response = []
        for question in request.questions:
            logging.info(f"Processing question: {question}")
            context_chunks = search_faiss_index(question, request.documents)
            logging.info(f"Found {len(context_chunks)} chunks from faiss index for question: {question}")
            response = ask_llm_with_context(query=question, context_chunks=context_chunks)
            logging.info("Returning response from LLM.")
            final_response.append(response)
        return {"answers": final_response}

    except Exception as e:
        logging.error(f"Error during hackrx run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
