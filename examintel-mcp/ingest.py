"""
ingest.py — Parses PYQ/syllabus PDFs from data/pdfs/<subject>/<year>.pdf,
chunks them (question-boundary heuristic with fallback to fixed windows),
embeds them with a local ONNX model (no API key, no cost, no internet needed
after the one-time model download), and stores everything in a local
ChromaDB collection that server.py queries.

Folder convention (this is how subject/year metadata gets derived — no
manual tagging needed):
    data/pdfs/COA/2022.pdf
    data/pdfs/COA/2023.pdf
    data/pdfs/CN/2022.pdf

Run this once to build the index, and again any time you add new PDFs:
    python ingest.py
"""

import os
import re
import pdfplumber
import chromadb
from fastembed import TextEmbedding
from html.parser import HTMLParser

# Regex to strip emoji and other non-BMP / problematic Unicode characters that
# break Windows cp1252 encoding on Claude Desktop's stdio transport.
_NON_ASCII_RE = re.compile(
    r'['
    r'\U00010000-\U0010ffff'   # supplementary planes (emoji, symbols)
    r'\u2700-\u27bf'           # dingbats
    r'\u2600-\u26ff'           # misc symbols
    r'\ufb00-\ufb4f'           # alphabetic presentation forms
    r'\ufe20-\ufe2f'           # combining half marks
    r'\u200b-\u200f'           # zero-width / directional
    r'\u2028-\u2029'           # line/paragraph separators
    r']+',
    flags=re.UNICODE,
)


def _sanitize_text(text: str) -> str:
    """Remove emoji and problematic Unicode so the resulting chunks can be
    safely printed on any Windows console / stdio transport."""
    return _NON_ASCII_RE.sub('', text)

_PROJECT_DIR = os.environ.get("EXAMINTEL_DATA_DIR") or os.path.dirname(os.path.abspath(__file__))
PDF_ROOT = os.path.join(_PROJECT_DIR, "data", "pdfs")
CHROMA_PATH = os.path.join(_PROJECT_DIR, "chroma_db")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # ~67MB, CPU-only, downloads once then runs fully offline

# Matches lines that look like the start of a numbered question: "Q1", "Q.1", "1.", "1)"
QUESTION_BOUNDARY = re.compile(r"^\s*(?:Q[\.\)]?\s*\d+[\.\)]?|\d{1,2}[\.\)])\s+", re.MULTILINE | re.IGNORECASE)


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
    def handle_data(self, data):
        self.text_parts.append(data)
    def get_text(self):
        return "".join(self.text_parts)


def extract_text(file_path: str) -> str:
    """Pull raw text out of a text-native PDF or HTML file."""
    if file_path.lower().endswith((".html", ".htm")):
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        parser = HTMLTextExtractor()
        parser.feed(html_content)
        return parser.get_text()
    
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def chunk_text(text: str, fallback_words: int = 180, overlap_words: int = 30) -> list[str]:
    """Try to split on question boundaries first (keeps each chunk as one real
    question, which makes search results readable). If fewer than 2 boundaries
    are found — different paper format, OCR noise, whatever — fall back to a
    fixed-size sliding window so nothing silently fails."""
    matches = list(QUESTION_BOUNDARY.finditer(text))
    if len(matches) >= 2:
        chunks = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk = text[start:end].strip()
            if len(chunk) > 15:
                chunks.append(chunk)
        if chunks:
            return chunks

    words = text.split()
    if not words:
        return []
    step = max(fallback_words - overlap_words, 1)
    return [
        " ".join(words[i:i + fallback_words])
        for i in range(0, len(words), step)
        if words[i:i + fallback_words]
    ]


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Delete and recreate to avoid stale chunks from removed/renamed files
    try:
        client.delete_collection("pyqs")
    except Exception:
        pass
    collection = client.get_or_create_collection("pyqs")
    print("Loading embedding model (first run downloads ~67MB, then it's cached)...")
    embedder = TextEmbedding(model_name=EMBED_MODEL)

    if not os.path.isdir(PDF_ROOT):
        print(f"No '{PDF_ROOT}' folder found. Create data/pdfs/<subject>/<year>.pdf and rerun.")
        return

    all_chunks, all_ids, all_metadatas = [], [], []
    counter = 0

    for subject in sorted(os.listdir(PDF_ROOT)):
        subject_dir = os.path.join(PDF_ROOT, subject)
        if not os.path.isdir(subject_dir):
            continue
        for fname in sorted(os.listdir(subject_dir)):
            if not fname.lower().endswith((".pdf", ".html", ".htm")):
                continue
            year = os.path.splitext(fname)[0]
            path = os.path.join(subject_dir, fname)
            print(f"Processing {subject}/{fname}...")
            text = _sanitize_text(extract_text(path))
            if not text.strip():
                print("  WARNING: no extractable text — likely a scanned PDF. "
                      "OCR fallback isn't wired up yet (see README). Skipping.")
                continue
            chunks = chunk_text(text)
            print(f"  -> {len(chunks)} chunks")
            for chunk in chunks:
                all_chunks.append(chunk)
                all_ids.append(f"chunk-{counter}")
                all_metadatas.append({"subject": subject, "year": year, "source": fname})
                counter += 1

    if not all_chunks:
        print("Nothing indexed. Add PDFs under data/pdfs/<subject>/<year>.pdf and rerun.")
        return

    print(f"Embedding {len(all_chunks)} chunks locally...")
    embeddings = [e.tolist() for e in embedder.embed(all_chunks)]

    collection.upsert(ids=all_ids, embeddings=embeddings, documents=all_chunks, metadatas=all_metadatas)
    print(f"Done. Indexed {len(all_chunks)} chunks into '{CHROMA_PATH}'.")
    print("Now run: python server.py   (or wire it into Claude Desktop per the README)")


if __name__ == "__main__":
    main()
