"""
Document Processor Module
Handles PDF parsing, text extraction, cleaning, and sliding-window chunking
with metadata tracking for RAG source attribution.
"""

from typing import List, Dict, Any, BinaryIO
import re
from pypdf import PdfReader


def extract_text_from_pdf(file_stream: BinaryIO, filename: str = "document.pdf") -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from an uploaded PDF stream.
    
    Returns:
        List of dicts: [{"page": 1, "text": "...", "source": filename}, ...]
    """
    reader = PdfReader(file_stream)
    pages_data = []

    for index, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        # Normalize whitespace while preserving paragraph breaks
        cleaned_text = re.sub(r"[ \t]+", " ", raw_text).strip()
        if cleaned_text:
            pages_data.append({
                "page": index + 1,
                "text": cleaned_text,
                "source": filename
            })

    return pages_data


def chunk_document(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Splits text across pages into overlapping chunks to ensure semantic continuity.
    
    Why Chunking?
    LLMs have context window limits, and embeddings work best when focused on
    coherent semantic units rather than entire documents.
    
    Why Overlap?
    Prevents splitting a vital sentence or idea across chunk boundaries, ensuring
    retrieval won't miss context that crosses a boundary.
    """
    chunks = []
    chunk_counter = 0

    for item in pages_data:
        text = item["text"]
        page_num = item["page"]
        source = item["source"]

        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # If not at the end of the text, try to break at sentence or word boundary
            if end < text_length:
                boundary_match = re.search(r"(\. |\n|\? |\! )", text[start:end])
                if boundary_match:
                    # Find last punctuation boundary in the latter half of chunk
                    last_break = max(
                        text.rfind(". ", start, end),
                        text.rfind("\n", start, end),
                        text.rfind("? ", start, end),
                        text.rfind("! ", start, end)
                    )
                    if last_break != -1 and last_break > start + (chunk_size // 2):
                        end = last_break + 1

            chunk_content = text[start:end].strip()
            if len(chunk_content) > 30:  # Ignore trivial or empty fragments
                chunk_counter += 1
                chunks.append({
                    "id": f"{source}_p{page_num}_c{chunk_counter}",
                    "text": chunk_content,
                    "metadata": {
                        "source": source,
                        "page": int(page_num),
                        "chunk_index": int(chunk_counter)
                    }
                })

            # Advance by step size (chunk_size - chunk_overlap)
            step = max(1, end - start - chunk_overlap)
            start += step

    return chunks
