"""Evaluation + ablation: proves your design choices with numbers.

eval set format (data/eval/eval_set.json):
  [{"question": "...", "evidence": ["exact phrase that must appear in a relevant chunk", ...]}]

Usage:  python -m rag.evaluate            # retrieval ablation
        python -m rag.evaluate --judge    # + LLM-as-judge faithfulness (needs LLM running)
"""
import argparse
import re
import unicodedata
import json
import time

import config
from rag import llm
from rag.pipeline import RAGPipeline
from rag.retrieve import MODES


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", s).lower())


def is_relevant(chunk, evidence):
    t = _norm(chunk["text"])
    return any(_norm(e) in t for e in evidence)


def retrieval_eval(pipe, data, k):
    print(f"\n{'mode':<16}{'Recall@'+str(k):<12}{'MRR':<8}{'avg ms':<8}")
    for mode in MODES:
        hits_at_k, rr, lat = 0, 0.0, 0.0
        for item in data:
            t = time.time()
            res = pipe.retriever.search(item["question"], mode=mode, k=k)
            lat += (time.time() - t) * 1000
            rank = next((i + 1 for i, c in enumerate(res) if is_relevant(c, item["evidence"])), None)
            if rank:
                hits_at_k += 1
                rr += 1 / rank
        n = len(data)
        print(f"{mode:<16}{hits_at_k/n:<12.3f}{rr/n:<8.3f}{lat/n:<8.0f}")


def faithfulness_eval(pipe, data):
    ok = 0
    for item in data:
        hits = pipe.retriever.search(item["question"])
        ctx = "\n".join(h["text"] for h in hits)
        ans = "".join(e["text"] for e in pipe.ask_stream(item["question"], use_expansion=False) if e["type"] == "token")
        verdict = llm.chat([{"role": "user", "content":
            f"Context:\n{ctx}\n\nAnswer:\n{ans}\n\nIs every claim in the answer supported by the context? "
            f"Reply with only YES or NO."}])
        time.sleep(5)
        ok += verdict.strip().upper().startswith("YES")
    print(f"\nFaithfulness (LLM judge): {ok}/{len(data)} = {ok/len(data):.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(config.BASE / "data/eval/eval_set.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--judge", action="store_true")
    a = ap.parse_args()
    data = json.load(open(a.file, encoding="utf-8"))
    pipe = RAGPipeline()
    retrieval_eval(pipe, data, a.k)
    if a.judge:
        faithfulness_eval(pipe, data)
