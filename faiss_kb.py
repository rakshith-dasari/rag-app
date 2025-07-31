import os
import pickle

from sentence_transformers import SentenceTransformer
from utils import clean_text, get_kb_name_from_url, parse_pdf
import faiss
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("faiss_kb.log"),
                        logging.StreamHandler()
                    ])

model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_kb(pdf_url: str):
    logging.info(f"Generating knowledge base for PDF: {pdf_url}")
    try:
        KB_NAME = get_kb_name_from_url(pdf_url)
        texts = parse_pdf(pdf_url)
        # Step 3: Generate embeddings using all-MiniLM
        logging.info("Generating embeddings...")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        logging.info("Embeddings generated.")

        # Step 4: Build FAISS index
        logging.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings)
        logging.info(f"FAISS index created with {faiss_index.ntotal} vectors.")

        # Step 5: Save index and metadata
        logging.info("Saving FAISS index and texts...")
        faiss.write_index(faiss_index, f"{KB_NAME}.index")
        with open(f"{KB_NAME}_texts.pkl", "wb") as f:
            pickle.dump(texts, f)

        logging.info(f"FAISS index and texts saved as {KB_NAME}.")
    except Exception as e:
        logging.error(f"Error generating knowledge base: {e}")
        raise

def search_faiss_index(query: str, pdf_url: str, top_k: int = 5) -> list[str]:
    """
    Search the FAISS index for the top_k most relevant text chunks based on the query.
    """
    logging.info(f"Searching FAISS index for query: '{query}'")
    try:
        # Load the FAISS index and texts
        index_name = get_kb_name_from_url(pdf_url)
        logging.info(f"Loading FAISS index: {index_name}.index")
        faiss_index = faiss.read_index(f"{index_name}.index")
        logging.info(f"Loading texts: {index_name}_texts.pkl")
        with open(f"{index_name}_texts.pkl", "rb") as f:
            texts = pickle.load(f)

        # Generate embedding for the query
        logging.info("Generating embedding for the query...")
        query_embedding = model.encode([query], convert_to_numpy=True)
        logging.info("Query embedding generated.")

        # Search the index
        logging.info(f"Searching index for top {top_k} results...")
        distances, indices = faiss_index.search(query_embedding, top_k)
        logging.debug(f"Distances: {distances}")
        logging.debug(f"Indices: {indices}")

        # Retrieve the corresponding text chunks
        context_chunks = [texts[i] for i in indices[0]]
        logging.info(f"Found {len(context_chunks)} relevant text chunks.")
        return context_chunks
    except Exception as e:
        logging.error(f"Error searching FAISS index: {e}")
        raise
