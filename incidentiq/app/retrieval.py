import re, numpy as np
from rank_bm25 import BM25Okapi
from .embed import get_embedder

TOK = re.compile(r"[A-Za-z0-9_\-\.:/]+")
def tokenize(t):
    toks = []
    for w in TOK.findall(t.lower()):
        toks.append(w)
        parts = [p for p in re.split(r"[_\-\.:/]", w) if p]
        if len(parts) > 1: toks.extend(parts)
    return toks

def rrf(rank_lists, k=60):
    s = {}
    for rl in rank_lists:
        for r, i in enumerate(rl):
            s[i] = s.get(i, 0) + 1 / (k + r + 1)
    return sorted(s, key=s.get, reverse=True), s

class Reranker:
    """Cross-encoder if EMBED_BACKEND=st, else lexical pair scorer (query-term coverage + exact-id boost)."""
    def __init__(self):
        self.ce = None
        import os
        if os.getenv("EMBED_BACKEND") == "st":
            try:
                from sentence_transformers import CrossEncoder
                self.ce = CrossEncoder("BAAI/bge-reranker-base")
            except Exception as e: print("cross-encoder unavailable:", e)
    def score(self, query, docs, error_ids=()):
        if self.ce:
            return list(self.ce.predict([(query, d["text"]) for d in docs]))
        q = set(tokenize(query)); out = []
        for d in docs:
            dt = set(tokenize(d["text"]))
            cov = len(q & dt) / (len(q) or 1)
            boost = 1.0 if d["meta"].get("error_id") in error_ids else 0.0
            out.append(cov + boost)
        return out

class Index:
    def __init__(self, docs):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])
        self.emb = get_embedder().fit([d["text"] for d in docs])
        self.vec = self.emb.encode([d["text"] for d in docs])
        self.reranker = Reranker()

    def _mask(self, f):
        f = f or {}
        ok = []
        for i, d in enumerate(self.docs):
            m = d["meta"]
            if all(d["type"] == "knowledge" or m.get(k) == v for k, v in f.items() if k in ("service", "environment", "level", "trace_id")):
                ok.append(i)
        return ok

    def search(self, rw, mode="full", k=5, pool=30):
        q = rw["rewritten"] if mode == "full" else rw["original"]
        f = rw["filters"] if mode == "full" else {}
        allowed = set(self._mask(f))
        bs = self.bm25.get_scores(tokenize(q))
        ds = self.vec @ self.emb.encode([q])[0]
        bl = [i for i in np.argsort(-bs)[:pool*3] if i in allowed and bs[i] > 0][:pool]
        dl = [i for i in np.argsort(-ds)[:pool*3] if i in allowed][:pool]
        if mode == "dense": order, sc = dl, {i: float(ds[i]) for i in dl}
        elif mode == "bm25": order, sc = bl, {i: float(bs[i]) for i in bl}
        else: order, sc = rrf([bl, dl])
        cand = order[:pool]
        bl = [int(i) for i in bl]; dl = [int(i) for i in dl]; cand = [int(i) for i in cand]
        trace = {"bm25_top": bl[:5], "dense_top": dl[:5], "fused_top": cand[:5]}
        if mode == "full" and cand:
            rs = self.reranker.score(q, [self.docs[i] for i in cand], rw["error_ids"])
            pairs = sorted(zip(rs, cand), key=lambda x: -x[0])
            cand = [c for _, c in pairs]
            sc = {c: float(r) for r, c in pairs}
            trace["reranked_top"] = cand[:5]
        hits = [{"doc": self.docs[i], "score": float(sc.get(i, 0)), "bm25": float(bs[i]), "dense": float(ds[i])} for i in cand[:k]]
        return hits, trace
