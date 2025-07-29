import re

def clean_text(text: str) -> str:
    text = text.strip()                           # Remove leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text)              # Collapse multiple whitespace/newlines
    text = re.sub(r'\.{3,}', '...', text)         # Normalize long ellipses
    text = re.sub(r'[^a-zA-Z0-9,.?!;:\-\'"()\s]', '', text)  # Remove weird symbols
    text = re.sub(r'(?<=\w)- (?=\w)', '', text)   # Fix hyphenation from PDF line breaks
    return text
