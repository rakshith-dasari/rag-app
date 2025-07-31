import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from utils import clean_text, parse_pdf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("pinecone_kb.log"),
                        logging.StreamHandler()
                    ])

# Ensure you have the necessary environment variables set
from dotenv import load_dotenv
load_dotenv()

def generate_pinecone_kb(pdf_path: str):
    logging.info(f"Generating Pinecone knowledge base for {pdf_path}")
    try:
        

        texts = parse_pdf(pdf_path)

        # Step 2: Initialize Pinecone
        logging.info("Initializing Pinecone...")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        index_name = "pdf-index"
        if index_name not in pc.list_indexes().names():
            logging.info(f"Creating new Pinecone index: {index_name}")
            pc.create_index_for_model(
                name=index_name,
                cloud="aws",  # or "gcp"
                region="us-east-1",  # adjust to your region
                embed={
                    "model": "llama-text-embed-v2",  # or another from Pinecone's model gallery
                    "field_map": {"text": "chunk_text"},
                    "metric": "cosine"
                }
            )
        else:
            logging.info(f"Using existing Pinecone index: {index_name}")

        index = pc.Index(index_name)

        # Step 3: Upsert chunks directly
        records = [
            {"id": f"chunk-{i}", "chunk_text": text}
            for i, text in enumerate(texts)
        ]

        def batch_records(records, batch_size=96):
            for i in range(0, len(records), batch_size):
                yield records[i:i+batch_size]

        logging.info("Upserting records to Pinecone...")
        for batch in batch_records(records, batch_size=96):
            index.upsert_records(namespace="__default__", records=batch)

        logging.info(f"Upserted {len(records)} text chunks to Pinecone using built-in embedding.")

    except Exception as e:
        logging.error(f"Error generating Pinecone knowledge base: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    generate_pinecone_kb("main.pdf")
