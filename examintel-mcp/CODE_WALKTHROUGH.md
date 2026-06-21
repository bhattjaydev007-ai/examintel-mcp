# ExamIntel MCP — Full Code Walkthrough

This covers every file, every import, and every function in the project — the
goal is that you can defend any line of this in an interview, not just point
at a working demo.

## 1. The architecture, in one paragraph

MCP (Model Context Protocol) is a standard way for an LLM client — Claude
Desktop, Cursor, anything that speaks the protocol — to call functions you
wrote, the same underlying idea as "function calling," just not tied to one
vendor. Three pieces: a **client** (Claude Desktop), a **server** (`server.py`,
the code that does the actual work), and a **transport** connecting them —
here, `stdio`, meaning the client launches `server.py` as a subprocess and the
two talk over stdin/stdout. The project is split into two scripts with two
different jobs on purpose: `ingest.py` runs occasionally (whenever you add
PDFs) to do the slow work — parsing, embedding — once; `server.py` runs every
time Claude needs an answer, and only ever reads what `ingest.py` already
built. Separating "build the index" from "answer questions using the index"
is a standard RAG pattern, not something specific to this project.

## 2. `ingest.py`

### Module docstring (lines 1–16)
Not just a comment — this is the first thing anyone (including you, in three
months) reads to understand what the file does and how to run it. The folder
convention it documents (`data/pdfs/<subject>/<year>.pdf`) is load-bearing:
the code below derives metadata from folder names, so this isn't optional
formatting, it's the actual data contract.

### Imports
```python
import os
import re
import pdfplumber
import chromadb
from fastembed import TextEmbedding
```
- **`os`** — filesystem operations: walking directories (`os.listdir`),
  building paths in an OS-independent way (`os.path.join`, so this works on
  both your Windows machine and Linux/Mac without changes), and reading the
  `EXAMINTEL_DATA_DIR` environment variable.
- **`re`** — Python's regex module, used for the question-boundary pattern
  that splits PDF text into individual questions.
- **`pdfplumber`** — the actual PDF text extraction library. Chosen over
  alternatives (PyPDF2, PyMuPDF) mainly for being simple to use for plain text
  pulls; it has no idea what to do with a scanned/image-only PDF, which is
  the documented OCR gap.
- **`chromadb`** — the vector database that stores chunks alongside their
  embeddings and metadata, and does the actual nearest-neighbor search later.
- **`from fastembed import TextEmbedding`** — the embedding model runner.
  Specifically *not* `sentence-transformers`, because that pulls in PyTorch,
  which pulls in multiple GB of CUDA packages even with no GPU present — this
  is the thing that filled up my test sandbox's disk during development.
  `fastembed` runs the same kind of model through ONNX Runtime instead: same
  job, a fraction of the install size.

### Constants
```python
_PROJECT_DIR = os.environ.get("EXAMINTEL_DATA_DIR") or os.path.dirname(os.path.abspath(__file__))
PDF_ROOT = os.path.join(_PROJECT_DIR, "data", "pdfs")
CHROMA_PATH = os.path.join(_PROJECT_DIR, "chroma_db")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
```
`_PROJECT_DIR` resolves to an **absolute** path, anchored either to an
explicit environment variable (used when this logic runs inside the packaged
extension) or to this file's own location on disk (`__file__`) when you run
it directly. This matters because a *relative* path like `"chroma_db"` means
"wherever the process happens to be started from" — fine when you run
`python ingest.py` from inside the folder, silently wrong the moment anything
else launches the script from a different working directory. `EMBED_MODEL` is
just the model name fastembed downloads and caches the first time it's used —
small (~67MB), CPU-only, no GPU required.

```python
QUESTION_BOUNDARY = re.compile(r"^\s*(?:Q[\.\)]?\s*\d+[\.\)]?|\d{1,2}[\.\)])\s+", re.MULTILINE | re.IGNORECASE)
```
This regex is the one piece of "clever" logic in the file, so it's worth
reading slowly. It matches the *start of a line* (`^` combined with
`re.MULTILINE`, so it checks every line, not just the start of the whole
text) that looks like a question number, in two possible shapes: `Q[\.\)]?\s*\d+[\.\)]?`
matches "Q1.", "Q.1", "Q1)" — an optional punctuation mark before the
digits, the digits themselves, then an optional punctuation mark after; or
`\d{1,2}[\.\)]` matches plain "1." or "1)" style numbering. Either way it has
to be followed by whitespace (`\s+`) before the actual question text. This
exact form exists because of a bug caught during testing: an earlier version
only allowed punctuation *before* the digits, so it matched "Q1" but not the
period in "Q1. Explain..." — it kept eating into the next character looking
for required whitespace that wasn't there yet. Real bug, real fix, worth
knowing the history of if you're asked "did you test this."

### `extract_text(pdf_path)`
```python
def extract_text(pdf_path: str) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)
```
Opens one PDF, walks every page, pulls whatever text `pdfplumber` can find on
each page. `page.extract_text() or ""` exists because `extract_text()`
returns `None` for a page with no text (a scanned image page, for instance) —
without the `or ""`, joining a list containing `None` would crash. Joining
with `"\n"` keeps page boundaries as line breaks rather than running every
page's text together into one unreadable blob.

### `chunk_text(text, fallback_words=180, overlap_words=30)`
```python
def chunk_text(text: str, fallback_words: int = 180, overlap_words: int = 30) -> list[str]:
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
```
Two strategies, tried in order. First: find every question-boundary match,
then treat the text *between* one match and the next as one chunk — that's
what `start = m.start()` / `end = matches[i+1].start()` is doing, with the
last match running to the end of the text (`len(text)`) since there's no
"next" match to stop at. The `len(chunk) > 15` check throws out near-empty
matches (a stray number that isn't really a question). This only kicks in if
at least 2 boundaries were found (`len(matches) >= 2`) — one match alone
isn't enough signal that this paper actually uses this numbering style.

If that produces nothing usable, it falls back to a **sliding window**:
split the whole text into words, then take overlapping groups of
`fallback_words` (180) with `overlap_words` (30) of overlap between
consecutive windows. The overlap exists so a sentence that would otherwise
get cut in half at a window boundary still appears whole in the next window.
`step = fallback_words - overlap_words` (150) is how far the window slides
each time — smaller than the window itself, which is what creates the
overlap. This path guarantees `chunk_text` never silently returns nothing for
real text, even if the paper's numbering doesn't match the regex at all.

### `main()`
Walks `PDF_ROOT` two levels deep: subject folders, then PDF files inside
each. `year = os.path.splitext(fname)[0]` strips ".pdf" off the filename to
get the year — this is the entire mechanism by which "2023.pdf" becomes the
metadata tag `{"year": "2023"}`, no manual tagging anywhere. For each PDF: extract
text, skip with a warning if empty (the OCR gap), chunk what's left, and
accumulate three parallel lists — `all_chunks` (the text), `all_ids`
(`"chunk-0"`, `"chunk-1"`, ...), `all_metadatas` (`{"subject", "year", "source"}`
per chunk). All three lists stay in the same order on purpose, since
ChromaDB's `upsert` call at the end zips them back together by position.

Embedding happens once, in a single batch (`embedder.embed(all_chunks)`)
after every PDF has been processed — embedding in one batch rather than
per-chunk is meaningfully faster, since the model can process several chunks
together instead of one at a time. `.tolist()` converts the model's native
array format into plain Python lists, which is what ChromaDB's API expects.
`collection.upsert(...)` writes everything to disk in one call — "upsert"
(update-or-insert) means re-running `ingest.py` after adding new PDFs updates
existing chunk IDs in place rather than duplicating them.

## 3. `server.py`

### Imports
```python
import os
import re
from collections import Counter

import chromadb
from fastembed import TextEmbedding
from mcp.server.fastmcp import FastMCP
```
`os` and `re` — same roles as in `ingest.py` (path handling; word-matching,
here for keyword extraction rather than question boundaries).
**`Counter`** from the standard library's `collections` module — a dictionary
subclass purpose-built for counting things and immediately gives you a
`.most_common(n)` method, which is exactly what `topic_frequency` needs;
writing that counting logic by hand would just be reinventing this. `chromadb`
and `fastembed` — same libraries as `ingest.py`, because this file has to
read the *same* index using the *same* embedding model `ingest.py` wrote it
with; embeddings from two different models aren't comparable to each other.
**`from mcp.server.fastmcp import FastMCP`** — the actual MCP server
framework. `FastMCP` is a decorator-based wrapper around the lower-level MCP
protocol implementation: instead of hand-writing the JSON-RPC message
handling the protocol requires, you write a plain Python function and one
decorator turns it into a properly-exposed tool.

### Data directory resolution and lazy loading
```python
_DATA_DIR = os.environ.get("EXAMINTEL_DATA_DIR") or os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(_DATA_DIR, "chroma_db")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

mcp = FastMCP("examintel")

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
```
`_DATA_DIR` mirrors `ingest.py`'s logic exactly, and has to: both files need
to agree on where `chroma_db` lives, or the server will look in an empty
folder for an index that's sitting somewhere else. `EXAMINTEL_DATA_DIR` is
what makes the packaged Claude Desktop extension work at all — when Claude
Desktop runs this file from its own extension storage (not from this project
folder), `__file__` would point to the wrong place; the environment variable,
set from the directory you pick during install, overrides that.

`mcp = FastMCP("examintel")` creates the server object — `"examintel"` is the
name that shows up in Claude Desktop's extension/connector list.

The `_collection = None` / `_embedder = None` plus the two `_get_*()`
functions are a **lazy-loading** pattern, and it exists to fix a real bug:
the embedding model used to load the moment this file was imported, which
meant the server's startup was blocked on a model download — and in a
network-restricted environment, it meant the server couldn't even start. Each
`_get_*()` function checks if the thing's already loaded; if not, loads it
*once* and remembers it (`global` lets the function modify the module-level
variable instead of creating a new local one); if it's already loaded, just
hands back the same object. The result: importing this file is now instant,
and the actual model/database connection only happens the first time a tool
is actually called.

```python
_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "and", "in", "for", "on", "with",
    "what", "explain", "define", "write", "short", "note", "notes", "marks",
    "describe", "discuss", "list", "state", "give", "any", "following", "briefly",
}
```
A plain set of words to ignore when counting term frequency — ordinary
English stopwords (the, a, of...) plus exam-paper-specific filler ("explain",
"marks", "briefly") that would otherwise dominate the frequency count without
telling you anything about actual exam *content*.

```python
def _embed_one(text: str) -> list[float]:
    return list(_get_embedder().embed([text]))[0].tolist()
```
A small wrapper because `fastembed`'s `.embed()` method is built to embed a
*batch* of texts at once (it takes a list, returns a list of vectors) — for
embedding a single search query, this wraps it in a one-item list, takes the
first (and only) result back out, and converts it to a plain list the way
`ingest.py` does.

### The three tools

```python
@mcp.tool()
def search_topic(subject: str, query: str, top_k: int = 5) -> str:
```
The `@mcp.tool()` decorator is what actually exposes this function to an LLM
client — it reads the function's name, its type-hinted parameters
(`subject: str`, `query: str`, `top_k: int = 5`), and its docstring, and
builds the schema the client receives describing what this tool does and how
to call it. **The docstring isn't documentation for humans reading the code —
it's the literal text the LLM reads** when deciding whether this tool is
relevant and what arguments to pass. That's why it reads like an instruction
("Semantic search across indexed past exam papers...") rather than a
one-line label.

Inside: embed the query the same way `ingest.py` embedded the stored chunks
(`_embed_one(query)`), then ask ChromaDB for the `top_k` nearest matches,
restricted to one subject (`where={"subject": subject}`). `results.get(...)[0]`
unwraps ChromaDB's response format, which nests results one level deeper
than you'd expect (it's built to handle multiple queries in one call, so even
a single query's results come back inside a one-item outer list). The rest
is formatting: pairing each returned chunk with its metadata and printing the
year and source filename next to it — this, concretely, is the entire
mechanism behind "results are citable."

```python
@mcp.tool()
def topic_frequency(subject: str, top_n: int = 10) -> str:
```
Deliberately *not* using embeddings at all. `_get_collection().get(where={"subject": subject})`
pulls every stored chunk for that subject (no similarity search, no query
vector — just a metadata filter), then `re.findall(r"[a-zA-Z]{4,}", doc.lower())`
extracts every word of 4+ letters from each chunk, lowercased, and a
`Counter` tallies them up after dropping stopwords. This is plain word
counting, not topic modeling — worth saying out loud if asked, since "ranks
recurring topics" can sound like more sophisticated NLP is happening than
actually is. Simple and explainable beat impressive-sounding and
indefensible.

```python
@mcp.tool()
def generate_study_plan(subject: str, days_remaining: int) -> str:
```
This tool calls `topic_frequency` directly, as a regular Python function call
(not through the MCP protocol — it's just one function calling another
function in the same file), asking for as many top terms as there are days
remaining, then wraps that list in a sentence of framing text. There's no
separate "planning" intelligence here; it's `topic_frequency` with different
formatting around it. Again, worth being upfront about this distinction if
asked how the "AI" decides the plan — it doesn't, the ranking logic does.

```python
if __name__ == "__main__":
    mcp.run()
```
Standard Python idiom: code under this guard only runs when the file is
executed directly (`python server.py`), not when it's imported by something
else (like this walkthrough's own testing, or any future code that wants to
reuse a function from this file without starting the server). `mcp.run()`
with no arguments defaults to `stdio` transport — it starts listening on
stdin/stdout for JSON-RPC messages from a client, which is why running this
by hand produces no visible output and looks "stuck." It isn't stuck. It's
correctly waiting for a client that, run this way, never arrives.

## 4. `requirements.txt`
```
mcp==1.27.2
chromadb==1.5.9
pdfplumber==0.11.10
fastembed==0.8.0
```
Exact pinned versions, not `mcp>=1.27` or unpinned — this guarantees you
install the precise versions this code was actually tested against, rather
than whatever happens to be newest (and possibly behaviorally different) on
the day you run `pip install`.

## 5. The `extension/` folder — packaging this as a Claude Desktop extension

Claude Desktop installs local MCP servers as Desktop Extensions (`.mcpb`
files), not through manual config file editing. `extension/src/server.py` is
a copy of the root `server.py` (they need to stay in sync by hand if you
change the logic — there's no automatic link between them).

### `extension/manifest.json`
```json
{
  "manifest_version": "0.4",
  "name": "examintel-mcp",
  "display_name": "ExamIntel",
  "version": "1.0.0",
  ...
  "server": {
    "type": "uv",
    "entry_point": "src/server.py",
    "mcp_config": {
      "command": "uv",
      "args": ["run", "--directory", "${__dirname}", "src/server.py"],
      "env": { "EXAMINTEL_DATA_DIR": "${user_config.data_directory}" }
    }
  },
  "user_config": {
    "data_directory": { "type": "directory", "title": "Project Data Folder", "required": true }
  },
  ...
}
```
`"type": "uv"` is the key decision here. The alternative — bundling Python
plus every dependency directly into the package — runs into a real,
documented limitation: `chromadb`, `fastembed`, and `mcp` (via `pydantic`)
all rely on *compiled* code under the hood, and compiled dependencies don't
bundle portably across machines. With `"type": "uv"`, Claude Desktop instead
installs dependencies fresh, on your actual machine, from `pyproject.toml` at
install time — which is also why the packed `.mcpb` file is 3.5KB: there are
no dependencies inside it, just the manifest, the dependency list, and one
source file.

`mcp_config.command`/`args` is the literal command Claude Desktop runs to
start the server: `uv run --directory <wherever-this-got-installed> src/server.py`.
`${__dirname}` and `${user_config.data_directory}` are template variables —
Claude Desktop substitutes the real install path and whatever folder you
picked during setup, respectively, before running the command. That second
substitution is what becomes the `EXAMINTEL_DATA_DIR` environment variable
`server.py` reads.

`user_config.data_directory` defines the "Project Data Folder" picker you
saw during install — `"type": "directory"` is what makes Claude Desktop show
a folder browser instead of a text field, and `"required": true` means
installation can't complete without it being set.

### `extension/pyproject.toml`
```toml
[project]
name = "examintel-mcp"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "mcp>=1.27.0",
    "chromadb>=1.5.0",
    "pdfplumber>=0.11.0",
    "fastembed>=0.8.0",
]
```
The dependency list `uv` actually installs at install time — note these use
`>=`, not the exact pins in `requirements.txt`. That's intentional: the root
`requirements.txt` is for *you*, reproducing an exact tested environment for
local development; `pyproject.toml` is for `uv` installing fresh on whatever
machine the extension ends up on, where pinning too tightly risks a version
conflict with something else `uv` needs to resolve.

## 6. Repo structure, and why some things aren't in it

```
examintel-mcp/
├── ingest.py / server.py / requirements.txt / README.md / .gitignore
├── data/pdfs/          — folder structure tracked, actual PDFs are not
├── extension/          — manifest.json, pyproject.toml, src/server.py
venv/, chroma_db/, __pycache__/, *.mcpb   — generated, not committed
```

`venv/` and `chroma_db/` are regenerable from `requirements.txt` and
`ingest.py` respectively — committing generated output alongside the source
that generates it is redundant and bloats the repo. The actual PYQ PDFs are
excluded the same way: anyone cloning this brings their own papers, so
there's no reason to ship yours (and PDFs are exactly the kind of binary
content that makes a git repo slow to clone for no benefit). The packed
`.mcpb` is a build artifact of `extension/`, the same logic as not committing
compiled output anywhere else — rebuild it with `mcpb pack extension/` rather
than tracking the binary.
