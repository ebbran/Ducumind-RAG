# DocuMind RAG — Hybrid Retrieval-Augmented QA with Evaluation

Chat with your PDFs/notes. Answers are grounded, cited `[1]`, streamed, and the retrieval
pipeline is **measured** (Recall@k / MRR ablation), not just assumed to work.

## Levels (build them in this order, commit each as a milestone)

| Level | What | Files |
|---|---|---|
| 1 Basic | PDF/TXT loading, sentence-aware chunking with overlap, MiniLM embeddings, FAISS, LLM answer | `ingest.py`, `index.py`, `llm.py` |
| 2 Intermediate | BM25 + dense hybrid search, Reciprocal Rank Fusion, cross-encoder reranking, page-level citations | `retrieve.py` |
| 3 Advanced | Conversational query rewriting, multi-query expansion, low-confidence guardrail (refuses instead of hallucinating), LRU cache, latency tracking, streaming SSE API, upload + live reindex, Docker | `pipeline.py`, `app.py` |
| 4 Proof | Ablation (dense vs BM25 vs hybrid vs hybrid+rerank) + LLM-as-judge faithfulness | `evaluate.py` |

## Architecture
```
question + history -> rewrite -> multi-query expand
   -> [FAISS dense] + [BM25 sparse] -> RRF fusion (top 20)
   -> cross-encoder rerank (top 5) -> confidence guardrail
   -> grounded prompt w/ numbered context -> streamed answer + sources
```

## Run
```bash
pip install -r requirements.txt
# Free local LLM:
ollama pull llama3.2            # then keep `ollama serve` running
# OR any OpenAI-compatible API (OpenAI, Groq, Gemini's OpenAI endpoint):
# export LLM_PROVIDER=openai OPENAI_API_KEY=... OPENAI_MODEL=gpt-4o-mini

cp your_pdfs/*.pdf data/docs/
python -m rag.ingest            # build index
python app.py                   # http://localhost:5000
```
Docker: `docker build -t documind . && docker run -p 5000:5000 -v $(pwd)/data:/app/data documind`

## Evaluate (do this — it's what makes the project stand out)
1. Write 20–30 questions in `data/eval/eval_set.json` with an `evidence` phrase from the right passage.
2. `python -m rag.evaluate --judge`
3. Paste YOUR real numbers into the table below and tune `CHUNK_SIZE`, `TOP_K`, `MIN_RERANK_SCORE`.

| mode | Recall@5 | MRR | avg ms |
|---|---|---|---|
| dense | | | |
| sparse | | | |
| hybrid | | | |
| hybrid_rerank | | | |

## Resume bullets (fill in after you measure)
- Built a hybrid RAG system (FAISS + BM25 with RRF, cross-encoder reranking) with cited, streamed answers via a Flask SSE API; Dockerized.
- Designed an evaluation harness (Recall@k, MRR, LLM-judge faithfulness); hybrid+rerank improved Recall@5 from **X%** to **Y%** over dense-only on a **N**-question set.
- Added query rewriting, multi-query expansion, and a confidence guardrail that reduces hallucinated answers.

## Interview talking points
Why hybrid beats dense on exact terms/IDs · why RRF needs no score normalization · bi-encoder vs cross-encoder
tradeoff (speed vs accuracy) · chunk size vs recall · how you detect "no answer" · what you'd add next
(Qdrant/pgvector, RAGAS, auth, document-level ACLs, async ingestion).
