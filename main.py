from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from generation import ask_llm_with_context
from faiss_kb import search_faiss_index, generate_kb
import logging
import time

from pinecone_kb import generate_pinecone_kb, search_pinecone_kb

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("main.log"),
                        logging.StreamHandler()
                    ])

load_dotenv()

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    pdf_url: str
    top_k: int = 5
    model: str = "faiss"  # Default to faiss, can be overridden

@app.post("/search")
def run_search(request: SearchRequest, authorization: str = Header(None)):
    start = time.perf_counter()
    if authorization != f"Bearer {os.getenv('HACKRX_API_KEY')}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        if request.model == "pinecone":
            logging.info(f"Received Pinecone search request with query: {request.query}")
            generate_pinecone_kb(request.pdf_url)  # Ensure KB is generated
            context_chunks = search_pinecone_kb(query=request.query, pdf_url=request.pdf_url, top_k=request.top_k)
        else:
            logging.info(f"Received FAISS search request with query: {request.query}")
            generate_kb(request.pdf_url)  # Ensure KB is generated
            context_chunks = search_faiss_index(query=request.query, pdf_url=request.pdf_url, top_k=request.top_k)
        logging.info("Returning response from LLM.")
        end = time.perf_counter()
        logging.info(f"Search completed in {end - start:.2f} seconds.")
        return context_chunks

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
def run_hackrx(request: HackRxRequest, authorization: str = Header(None)):
    start = time.perf_counter()
    if authorization != f"Bearer {os.getenv('HACKRX_API_KEY')}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        logging.info(f"Received hackrx request with {len(request.questions)} questions.")
        generate_kb(request.documents)
        # generate_pinecone_kb(request.documents) # pinecone
        logging.info(f"Knowledge base generated for {request.documents}")
        final_response = []
        for question in request.questions:
            logging.info(f"Processing question: {question}")
            context_chunks = search_faiss_index(question, request.documents)
            # context_chunks = search_pinecone_kb(query=question, pdf_url=request.documents) # pinecone
            logging.info(f"Found {len(context_chunks)} chunks from faiss index for question: {question}")
            response = ask_llm_with_context(query=question, context_chunks=context_chunks)
            logging.info("Returning response from LLM.")
            final_response.append(response)
        
        end = time.perf_counter()
        logging.info(f"HackRx run completed in {end - start:.2f} seconds.")
        logging.info(f"Final response: {final_response}")
        return {"answers": final_response}

    except Exception as e:
        logging.error(f"Error during hackrx run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
