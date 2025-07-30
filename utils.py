import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("utils.log"),
                        logging.StreamHandler()
                    ])

def clean_text(text: str) -> str:
    logging.debug(f"Cleaning text: '{text[:50]}...' ")
    text = text.strip()                           # Remove leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text)              # Collapse multiple whitespace/newlines
    text = re.sub(r'\.{3,}', '...', text)         # Normalize long ellipses
    text = re.sub(r'[^a-zA-Z0-9,.?!;:\-\'"()\s]', '', text)  # Remove weird symbols
    text = re.sub(r'(?<=\w)- (?=\w)', '', text)   # Fix hyphenation from PDF line breaks
    logging.debug(f"Cleaned text: '{text[:50]}...' ")
    return text
