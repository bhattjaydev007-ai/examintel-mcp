"""
server.py — The MCP server itself. Exposes exam-intelligence tools backed by
the ChromaDB index that ingest.py builds. Run ingest.py at least once before
using this. No API keys, no internet needed at query time — only the
one-time embedding model download (handled automatically on first run).

This file is also copied into extension/src/server.py and packaged as a
Claude Desktop extension (.mcpb) — see README.md for the install steps via
Settings > Extensions, not manual config file editing.
"""

import os
import re
from collections import Counter

import chromadb
from fastembed import TextEmbedding
from mcp.server.fastmcp import FastMCP

# EXAMINTEL_DATA_DIR lets the packaged Claude Desktop extension point at your
# existing project folder (the one with data/pdfs and chroma_db) via a
# user-selected directory. Falls back to this file's own location when run
# directly (e.g. python server.py during local development).
_DATA_DIR = os.environ.get("EXAMINTEL_DATA_DIR") or os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(_DATA_DIR, "chroma_db")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

mcp = FastMCP("examintel")

# Lazy-loaded on first use, not at import time — keeps server startup instant
# and avoids any network call until a tool is actually invoked.
_collection = None
_embedder = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection("pyqs")
    return _collection


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "and", "in", "for", "on", "with",
    "what", "explain", "define", "write", "short", "note", "notes", "marks",
    "describe", "discuss", "list", "state", "give", "any", "following", "briefly",
}


def _embed_one(text: str) -> list[float]:
    return list(_get_embedder().embed([text]))[0].tolist()


@mcp.tool()
def search_topic(subject: str, query: str, top_k: int = 5) -> str:
    """Semantic search across indexed past exam papers for a given subject and
    topic/question. Every result is tagged with its source year and filename,
    so answers are citable instead of just plausible-sounding text.

    Args:
        subject: subject folder name as organized under data/pdfs, e.g. "COA" or "CN"
        query: the topic or question to search for, e.g. "pipelining hazards"
        top_k: number of results to return (default 5)
    """
    results = _get_collection().query(
        query_embeddings=[_embed_one(query)],
        where={"subject": subject},
        n_results=top_k,
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        return (f"No results for subject '{subject}'. Check that this matches a "
                 f"folder name under data/pdfs and that you've run ingest.py.")

    lines = [f"Top {len(docs)} matches for '{query}' in {subject}:\n"]
    for doc, meta in zip(docs, metas):
        lines.append(f"[{meta.get('year', '?')} | {meta.get('source', '?')}]\n{doc.strip()}\n")
    return "\n".join(lines)


@mcp.tool()
def topic_frequency(subject: str, top_n: int = 10) -> str:
    """Rank recurring keywords/topics by how often they appear across all
    indexed years for a subject — a quick signal of what's repeatedly tested.

    Args:
        subject: subject folder name, e.g. "COA"
        top_n: how many top terms to return
    """
    data = _get_collection().get(where={"subject": subject})
    docs = data.get("documents", [])
    if not docs:
        return f"No indexed data for subject '{subject}'. Run ingest.py first."

    counts = Counter()
    for doc in docs:
        for word in re.findall(r"[a-zA-Z]{4,}", doc.lower()):
            if word not in _STOPWORDS:
                counts[word] += 1

    top = counts.most_common(top_n)
    lines = [f"Most frequent terms across indexed {subject} papers:\n"]
    lines += [f"  {term}: {count} occurrences" for term, count in top]
    return "\n".join(lines)


@mcp.tool()
def generate_study_plan(subject: str, days_remaining: int) -> str:
    """Generate a simple revision priority order based on historical topic
    frequency and how many days are left before the exam.

    Args:
        subject: subject folder name, e.g. "COA"
        days_remaining: how many days until the exam
    """
    n = max(days_remaining, 1)
    freq_block = topic_frequency(subject, top_n=n)
    return (
        f"Study plan for {subject} — {days_remaining} day(s) remaining:\n\n"
        f"Work through these in order, highest historical frequency first "
        f"(roughly one per focus block):\n\n{freq_block}\n\n"
        f"Use search_topic on any term above to pull the actual question variants."
    )


if __name__ == "__main__":
    mcp.run()
