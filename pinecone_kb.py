import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from utils import clean_text
# Ensure you have the necessary environment variables set
from dotenv import load_dotenv
load_dotenv()

# Step 1: Load and split PDF
loader = PyPDFLoader("main.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.split_documents(pages)

texts = [clean_text(doc.page_content) for doc in documents]

print(f"Loaded and cleaned {len(texts)} text chunks.")

# Step 2: Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "pdf-index"
if index_name not in pc.list_indexes().names():
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

index = pc.Index(index_name)

# Step 3: Upsert chunks directly
records = [
    {"id": f"chunk-{i}", "chunk_text": text}
    for i, text in enumerate(texts)
]

def batch_records(records, batch_size=96):
    for i in range(0, len(records), batch_size):
        yield records[i:i+batch_size]

for batch in batch_records(records, batch_size=96):
    index.upsert_records(namespace="__default__", records=batch)


print(f"Upserted {len(records)} text chunks to Pinecone using built-in embedding.")
