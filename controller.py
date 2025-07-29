from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from generation import ask_llm_with_context
from faiss_kb import search_faiss_index

load_dotenv()

app = FastAPI()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pdf-index"
NAMESPACE = "__default__"

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(INDEX_NAME)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def run_search(request: SearchRequest):
    try:
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

        return ask_llm_with_context(query=request.query, context_chunks=context_chunks)


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/faiss")
def run_faiss_search(request: SearchRequest):
    try:
        context_chunks = search_faiss_index(request.query, request.top_k)
        return ask_llm_with_context(query=request.query, context_chunks=context_chunks)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))