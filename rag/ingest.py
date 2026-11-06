"""LEVEL 1 - Ingestion: load -> clean -> chunk (sentence-aware, with overlap) -> index."""
import re
from pathlib import Path
from pypdf import PdfReader

import config

SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def load_documents(docs_dir=config.DOCS_DIR):
    pages = []
    for path in sorted(Path(docs_dir).rglob("*")):
        ext = path.suffix.lower()
        if ext == ".pdf":
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"source": path.name, "page": i, "text": text})
        elif ext in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                pages.append({"source": path.name, "page": 1, "text": text})
    return pages


def chunk_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    text = re.sub(r"[ \t]+", " ", text)
    units = []
    for u in SPLIT.split(text):
        u = (u or "").strip()
        if not u:
            continue
        while len(u) > size:          # hard-split very long sentences
            units.append(u[:size])
            u = u[size:]
        units.append(u)

    chunks, cur, cur_len = [], [], 0
    for u in units:
        if cur and cur_len + len(u) + 1 > size:
            chunks.append(" ".join(cur))
            keep, length = [], 0      # carry trailing sentences as overlap
            for s in reversed(cur):
                if length + len(s) > overlap:
                    break
                keep.insert(0, s)
                length += len(s) + 1
            cur, cur_len = keep, length
        cur.append(u)
        cur_len += len(u) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def build_chunks(pages):
    chunks = []
    for p in pages:
        for t in chunk_text(p["text"]):
            chunks.append({"id": len(chunks), "text": t, "source": p["source"], "page": p["page"]})
    return chunks


def build_index():
    from rag.index import Index
    pages = load_documents()
    chunks = build_chunks(pages)
    if not chunks:
        print("No documents found in", config.DOCS_DIR)
        return 0
    idx = Index()
    idx.build(chunks)
    idx.save()
    print(f"Indexed {len(chunks)} chunks from {len({c['source'] for c in chunks})} files.")
    return len(chunks)


if __name__ == "__main__":
    build_index()
