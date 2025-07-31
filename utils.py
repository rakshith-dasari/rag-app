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

# def parse_pdf(pdf_url:str) -> list[str]:
#     logging.info(f"Generating knowledge base for PDF: {pdf_url}")
#     logging.info("Loading and chunking PDF...")
#     loader = PyPDFLoader(pdf_url)
#     pages = loader.load()

#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     documents = splitter.split_documents(pages)

#     texts = [clean_text(doc.page_content) for doc in documents]
#     logging.info(f"Loaded and cleaned {len(texts)} text chunks.")
#     return texts

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

import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def split_with_headings(text: str,
                        heading_regex: str = r"^\d+\.\s+[A-Z].+$",
                        chunk_size: int = 800,
                        chunk_overlap: int = 100) -> list[str]:
    """
    Split raw text into chunks that each start with the last seen heading.
    """
    chunks, current_heading, buffer = [], None, []

    for line in text.splitlines():
        if re.match(heading_regex, line):
            current_heading = line.strip()
            continue

        buffer.append(line)
        # flush buffer if it’s getting large
        if sum(len(l) for l in buffer) >= chunk_size:
            body = "\n".join(buffer).strip()
            if body:
                chunks.append(f"{current_heading or ''}\n{body}")
            buffer = []

    # final flush
    if buffer:
        body = "\n".join(buffer).strip()
        chunks.append(f"{current_heading or ''}\n{body}")

    # second-level splitting for long bodies
    rsplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    final_chunks = []
    for chunk in chunks:
        final_chunks.extend(rsplitter.split_text(chunk))

    return final_chunks

from langchain.document_loaders import PyPDFLoader

def parse_pdf(pdf_url: str) -> list[str]:
    """
    1. Load the PDF
    2. Concatenate all pages into one string (so headings flow correctly)
    3. Split with heading-aware splitter
    4. Clean and return
    """
    logging.info(f"Generating knowledge base for PDF: {pdf_url}")

    # 1️⃣  Load pages
    loader = PyPDFLoader(pdf_url)
    pages = loader.load()
    full_text = "\n".join(p.page_content for p in pages)

    # 2️⃣  Heading-aware split
    logging.info("Chunking with heading-aware splitter ...")
    raw_chunks = split_with_headings(full_text,
                                     heading_regex=r"^\d+\.\s+[A-Z].+$",
                                     chunk_size=800,
                                     chunk_overlap=100)

    # 3️⃣  Clean each chunk (your existing clean_text util)
    texts = [clean_text(chunk) for chunk in raw_chunks]

    logging.info(f"Loaded and cleaned {len(texts)} text chunks.")
    return texts
