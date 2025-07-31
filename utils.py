import re
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("utils.log"),
                        logging.StreamHandler()
                    ])

def clean_text(text: str) -> str:
    logging.debug(f"Cleaning text: '{text[:50]}...' ")
    # footer_line_count = 6  # Number of footer lines to remove
    # # Step 1: Strip footer lines (from bottom)
    # lines = text.splitlines()
    # if len(lines) > footer_line_count:
    #     lines = lines[:-footer_line_count]
    
    # text = "\n".join(lines)
    text = text.strip()                           # Remove leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text)              # Collapse multiple whitespace/newlines
    text = re.sub(r'\.{3,}', '...', text)         # Normalize long ellipses
    text = re.sub(r'[^a-zA-Z0-9,.?!;:\-\'"()\s]', '', text)  # Remove weird symbols
    text = re.sub(r'(?<=\w)- (?=\w)', '', text)   # Fix hyphenation from PDF line breaks
    logging.debug(f"Cleaned text: '{text[:50]}...' ")
    return text

def parse_pdf(pdf_url:str) -> list[str]:
    logging.info(f"Generating knowledge base for PDF: {pdf_url}")
    logging.info("Loading and chunking PDF...")
    loader = PyPDFLoader(pdf_url)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = splitter.split_documents(pages)

    texts = [clean_text(doc.page_content) for doc in documents]
    logging.info(f"Loaded and cleaned {len(texts)} text chunks.")
    return texts

def get_kb_name_from_url(pdf_url: str) -> str:
    """
    Extract the knowledge base name from the PDF URL.
    """
    logging.info(f"Extracting knowledge base name from URL: {pdf_url}")
    base_name = pdf_url.split("/")[-1].replace(".pdf", "")
    # Create a hash for uniqueness
    hash_part = hashlib.md5(pdf_url.encode()).hexdigest()[:8]
    # Keep only lowercase alphanumeric characters in base_name
    base_name = re.sub(r'[^a-z0-9]', '', base_name.lower())
    kb_name = f"{base_name[:10]}-{hash_part}"
    logging.info(f"Knowledge base name: {kb_name}")
    return kb_name