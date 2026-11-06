# DocuMind RAG: Hybrid Retrieval-Augmented QA with Evaluation

Chat with your PDFs and notes. Answers are grounded in the retrieved passages, cited with `[1]`-style
markers, streamed token by token, and the retrieval pipeline is **measured** (Recall@k and MRR ablation)
rather than assumed to work.

## Features by level

| Level | What it adds | Files |
|---|---|---|
| 1. Basic | PDF/TXT loading, sentence-aware chunking with overlap, MiniLM embeddings, FAISS index, LLM answer | `rag/ingest.py`, `rag/index.py`, `rag/llm.py` |
| 2. Intermediate | BM25 + dense hybrid search, Reciprocal Rank Fusion, cross-encoder reranking, page-level citations | `rag/retrieve.py` |
| 3. Advanced | Conversational query rewriting, multi-query expansion, low-confidence guardrail (declines instead of guessing), LRU cache, latency tracking, streaming Flask API with chat UI, live upload and reindex, Docker | `rag/pipeline.py`, `app.py` |
| 4. Evaluation | Ablation across dense, sparse, hybrid and hybrid+rerank, plus an optional LLM-as-judge faithfulness check (`--judge`) | `rag/evaluate.py` |

## Architecture
```
question + chat history -> rewrite -> multi-query expansion
   -> [FAISS dense] + [BM25 sparse] -> RRF fusion (top 20)
   -> cross-encoder rerank (top 5) -> confidence guardrail
   -> grounded prompt with numbered context -> streamed answer + source cards
```

## Setup
```bash
pip install -r requirements.txt

# LLM option A: any OpenAI-compatible API (OpenAI, Groq, Gemini's OpenAI endpoint)
export LLM_PROVIDER=openai
export OPENAI_BASE_URL=https://api.groq.com/openai/v1
export OPENAI_API_KEY=<your key>
export OPENAI_MODEL=openai/gpt-oss-20b

# LLM option B: local Ollama
# ollama pull llama3.2   (LLM_PROVIDER=ollama is the default)

cp your_pdfs/*.pdf data/docs/
python -m rag.ingest            # build the index
python app.py                   # http://localhost:5000
```
Docker: `docker build -t documind . && docker run -p 5000:5000 -v $(pwd)/data:/app/data documind`

Never commit API keys. Keep them in environment variables or a git-ignored file.

## Evaluation

**Setup:** 30 questions over a corpus of 4 research papers on AI in medicine (301 chunks).
A retrieved chunk counts as relevant if it contains a reference phrase taken from the paper's text.
Metrics are computed over the top 5 results. Run it with `python -m rag.evaluate`.

| Mode | Recall@5 | MRR | Avg latency per query* |
|---|---|---|---|
| Dense (FAISS) | 0.900 | 0.697 | ~330 ms |
| Sparse (BM25) | 0.833 | 0.750 | ~1 ms |
| Hybrid (dense + BM25, RRF) | 0.867 | 0.808 | ~17 ms |
| **Hybrid + rerank** | **0.933** | **0.840** | ~2.0 s |

\*CPU only, measured after model warm-up on a GitHub Codespace. The first query is slower because models load.

### Comparison with the baselines

| Hybrid + rerank vs. | Recall@5 change | MRR change |
|---|---|---|
| Dense only | +0.033 | +0.143 (about +20% relative) |
| Sparse only | +0.100 | +0.090 |
| Hybrid (no rerank) | +0.067 | +0.032 |

### How to read these numbers
- Hybrid + rerank gives the best Recall@5 and the best MRR, so the right passage is both found more often
  and ranked higher.
- The gain comes at a cost: reranking adds roughly 2 seconds per query on CPU. Hybrid without reranking
  is a good option when latency matters.
- Relevance is judged by exact phrases from the source text, which favours keyword matching. That is
  likely why BM25 scores a higher MRR than dense retrieval here. Adding paraphrased questions would give
  dense retrieval a fairer test.
- With 30 questions, a single question changes Recall@5 by 0.033, so small gaps are not conclusive.

## Design notes
- **Why hybrid:** dense search handles paraphrase; BM25 handles exact terms, numbers and names.
- **Why RRF:** it fuses rankings without having to normalise scores from different retrievers.
- **Why a cross-encoder:** it scores the query and passage together, which is more accurate than
  comparing embeddings, but too slow to run over the whole corpus, so it only reranks the top 20.
- **Guardrail:** if the best reranker score is below `MIN_RERANK_SCORE`, the system says it could not
  find an answer instead of generating one.
- **Chunking:** about 800 characters with 150 overlap, split on sentence boundaries.

## Possible extensions
Qdrant or pgvector as the vector store, RAGAS metrics, paraphrased evaluation questions, authentication,
per-document access control, asynchronous ingestion, and a larger multi-domain corpus.
