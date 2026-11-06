"""LEVEL 3 - Full pipeline: query rewriting, multi-query expansion, hybrid+rerank retrieval,
grounded generation with citations, low-confidence guardrail, caching, latency tracking."""
import time
from collections import OrderedDict

import config
from rag import llm
from rag.index import Index
from rag.retrieve import Retriever

SYSTEM = (
    "You are DocuMind, a precise assistant. Answer ONLY from the numbered context below. "
    "Cite every claim with its source number like [1] or [2][3]. "
    "If the context does not contain the answer, say you don't know. Be concise."
)
NO_ANSWER = "I couldn't find a reliable answer to that in the uploaded documents."


class RAGPipeline:
    def __init__(self):
        self.cache = OrderedDict()
        self.reload()

    def reload(self):
        self.index = Index()
        self.ready = self.index.load()
        self.retriever = Retriever(self.index)
        self.cache.clear()

    # ---- query understanding -------------------------------------------------
    def rewrite(self, question, history):
        if not history:
            return question
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        prompt = (f"Conversation:\n{convo}\n\nRewrite the final user question as a standalone "
                  f"search query. Output only the query.\nQuestion: {question}")
        try:
            out = llm.chat([{"role": "user", "content": prompt}]).strip().strip('"')
            return out or question
        except Exception:
            return question

    def expand(self, question, n=2):
        prompt = (f"Give {n} alternative phrasings of this search query, one per line, "
                  f"no numbering, no extra text.\nQuery: {question}")
        try:
            raw = llm.chat([{"role": "user", "content": prompt}]).splitlines()
            lines = [l.strip(" -*0123456789.") for l in raw]
            return [question] + [l for l in lines if l][:n]
        except Exception:
            return [question]

    # ---- main entry -----------------------------------------------------------
    def ask_stream(self, question, history=None, use_expansion=True):
        history = history or []
        if not self.ready:
            yield {"type": "token", "text": "No documents indexed yet. Upload a PDF first."}
            yield {"type": "done", "timings": {}}
            return

        t0 = time.time()
        standalone = self.rewrite(question, history)
        queries = self.expand(standalone) if use_expansion else [standalone]
        yield {"type": "meta", "standalone_query": standalone, "queries": queries}

        key = standalone.lower().strip()
        if key in self.cache:
            hits, answer = self.cache[key]
            self.cache.move_to_end(key)
            yield {"type": "sources", "sources": self._fmt(hits)}
            yield {"type": "token", "text": answer}
            yield {"type": "done", "timings": {"cached": True}}
            return

        hits = self.retriever.search(queries, question=standalone)
        t_ret = time.time() - t0

        if not hits or hits[0]["score"] < config.MIN_RERANK_SCORE:   # guardrail
            yield {"type": "sources", "sources": []}
            yield {"type": "token", "text": NO_ANSWER}
            yield {"type": "done", "timings": {"retrieval_s": round(t_ret, 2)}}
            return

        yield {"type": "sources", "sources": self._fmt(hits)}
        context = "\n\n".join(f"[{i+1}] ({h['source']}, p.{h['page']})\n{h['text']}" for i, h in enumerate(hits))
        messages = [{"role": "system", "content": f"{SYSTEM}\n\nContext:\n{context}"}]
        messages += history[-6:] + [{"role": "user", "content": question}]

        parts = []
        for tok in llm.stream_chat(messages):
            parts.append(tok)
            yield {"type": "token", "text": tok}

        self.cache[key] = (hits, "".join(parts))
        if len(self.cache) > 128:
            self.cache.popitem(last=False)
        yield {"type": "done", "timings": {"retrieval_s": round(t_ret, 2), "total_s": round(time.time() - t0, 2)}}

    @staticmethod
    def _fmt(hits):
        return [{"id": i + 1, "source": h["source"], "page": h["page"],
                 "score": round(h["score"], 3), "snippet": h["text"][:300]} for i, h in enumerate(hits)]
