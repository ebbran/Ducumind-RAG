"""LEVEL 2 - Hybrid retrieval (dense + BM25) fused with Reciprocal Rank Fusion, then cross-encoder reranking."""
import config

MODES = ("dense", "sparse", "hybrid", "hybrid_rerank")


def rrf(rank_lists, k=60):
    fused = {}
    for lst in rank_lists:
        for rank, (idx, _) in enumerate(lst):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])


class Retriever:
    def __init__(self, index):
        self.index = index
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(config.RERANK_MODEL)
        return self._reranker

    def search(self, queries, question=None, mode="hybrid_rerank", k=config.TOP_K_FINAL):
        """queries: str or list[str] (multi-query). question: text used for reranking."""
        if isinstance(queries, str):
            queries = [queries]
        question = question or queries[0]
        pool = config.TOP_K_RETRIEVE
        lists = []
        for q in queries:
            if mode in ("dense", "hybrid", "hybrid_rerank"):
                lists.append(self.index.dense(q, pool))
            if mode in ("sparse", "hybrid", "hybrid_rerank"):
                lists.append(self.index.sparse(q, pool))
        fused = rrf(lists)[:pool]
        chunks = self.index.chunks

        if mode == "hybrid_rerank" and fused:
            cand = [i for i, _ in fused]
            scores = self.reranker.predict([(question, chunks[i]["text"]) for i in cand])
            ranked = sorted(zip(cand, scores), key=lambda x: -x[1])[:k]
            return [{**chunks[i], "score": float(s)} for i, s in ranked]
        return [{**chunks[i], "score": float(s)} for i, s in fused[:k]]
