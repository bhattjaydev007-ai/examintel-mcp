# examintel-mcp

An MCP (Model Context Protocol) server that turns your own PYQ/syllabus PDFs and HTML documents into
a queryable exam-intelligence tool — usable from Claude Desktop or any MCP client.
Runs fully locally. No API keys, no signups, no per-query cost.

## 1. Prerequisites

Python 3.10+ installed. Check with:
```
python3 --version
```
(On Windows, this is usually just `python --version` in Command Prompt or PowerShell.)

## 2. Setup

```bash
# from inside the examintel-mcp folder
python3 -m venv venv

# activate it:
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows (Command Prompt or PowerShell)

pip install -r requirements.txt
```

## 3. Add your PYQs

Organize PDFs or HTML files under `data/pdfs/<subject>/<year>.pdf` or `data/pdfs/<subject>/<source_name>.html` — the folder name becomes the
subject tag, and the filename (without extension) becomes the year/source tag. Example:

```
data/pdfs/COA/2022.pdf
data/pdfs/COA/2023.pdf
data/pdfs/COA/2024.pdf
data/pdfs/CN/CN_Master_Guide.html
data/pdfs/JAVA/Java_Master_Guide.html
```

Start with 2 subjects, not your whole semester — get the pipeline working end to
end first, then scale up.

## 4. Build the index

```bash
python ingest.py
```

First run downloads a small (~67MB) embedding model from Hugging Face — that's the
only time this needs internet. Every run after that, and every query, is fully
offline. You'll see a per-file chunk count printed; if a file shows 0 chunks, it's
probably a scanned/image-only PDF (see Limitations below).

## 5. Connect it to Claude Desktop

Claude Desktop installs local MCP servers as "Desktop Extensions" (`.mcpb` files), not through manual JSON config editing.

1. Open Claude Desktop → Settings → Extensions → Advanced settings → **Extension Developer** section → "Install Extension..."
2. Select the `.mcpb` file built from `extension/`.
3. When prompted for **Project Data Folder**, select this project's folder (the one with `data/pdfs` and `chroma_db`).
4. Restart Claude Desktop, then ask something like: "Use examintel to find COA pipelining questions"

## 6. The three tools

- **search_topic(subject, query)** — semantic search across indexed papers, returns matches with year + source
- **topic_frequency(subject)** — ranks recurring terms by how often they appear
- **generate_study_plan(subject, days_remaining)** — combines frequency with your timeline into a revision order

## 7. Limitations (be upfront about these in interviews — it reads better than pretending they don't exist)

Scanned/photocopied PDFs with no embedded text won't extract anything — OCR
fallback (pytesseract) isn't wired in yet; that's a clean, well-scoped v2 feature
to mention if asked "what would you add next." The question-boundary chunker is a
regex heuristic tuned for "Q1.", "Q.1", "1.", "1)" style numbering — unusual paper
formats may fall back to fixed-window chunking, which still works but loses the
"one chunk = one question" cleanliness. `topic_frequency` is keyword counting, not
true topic modeling — a fast, transparent first version, not a final one.

## 8. Next step: publish it

Once this works locally, publish to the official MCP registry
(`modelcontextprotocol/registry` on GitHub, via the `mcp-publisher` CLI) and
cross-list on mcp.so / smithery.ai. That's what turns this from "a project on my
laptop" into a public, clickable artifact — see the PRD for the full checklist.
