import os
from pathlib import Path

BASE = Path(__file__).parent
DOCS_DIR = BASE / "data" / "docs"
INDEX_DIR = BASE / "data" / "index"

# Models
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Chunking / retrieval
CHUNK_SIZE = 800        # characters
CHUNK_OVERLAP = 150
TOP_K_RETRIEVE = 20     # candidates per retriever before fusion
TOP_K_FINAL = 5         # chunks sent to the LLM
# Cross-encoder logits: > 0 is usually relevant. Tune on your eval set.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-3.0"))

# LLM: "ollama" (free, local) or "openai" (any OpenAI-compatible API, incl. Gemini/Groq)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
