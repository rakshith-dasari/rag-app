import os
import pickle
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from utils import clean_text
import faiss
model = SentenceTransformer("all-MiniLM-L6-v2")

def create_faiss_index():
    KB_NAME = "faiss_index"
    # Step 1: Load and chunk PDF
    loader = PyPDFLoader("main.pdf")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = splitter.split_documents(pages)

    texts = [clean_text(doc.page_content) for doc in documents]
    print(f"Loaded and cleaned {len(texts)} text chunks.")

    # Step 2: Generate embeddings using all-MiniLM
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Step 3: Build FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    print(f"FAISS index created with {faiss_index.ntotal} vectors.")

    # Step 4: Save index and metadata
    faiss.write_index(faiss_index, f"{KB_NAME}.index")
    with open(f"{KB_NAME}_texts.pkl", "wb") as f:
        pickle.dump(texts, f)

    print(f"FAISS index and texts saved as {KB_NAME}.")


def search_faiss_index(index_name: str, query: str, top_k: int = 5) -> list[str]:
    """
    Search the FAISS index for the top_k most relevant text chunks based on the query.
    """
    # Load the FAISS index and texts
    faiss_index = faiss.read_index(f"{index_name}.index")
    with open(f"{index_name}_texts.pkl", "rb") as f:
        texts = pickle.load(f)

    # Generate embedding for the query
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Search the index
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    # Retrieve the corresponding text chunks
    context_chunks = [texts[i] for i in indices[0]]
    
    return context_chunks