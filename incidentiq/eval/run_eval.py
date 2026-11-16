"""Compares 3 configs: dense-only, BM25+dense+RRF, rewrite+hybrid+rerank. Retrieval only (no LLM calls, zero API cost)."""
import os, sys, json, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["GROQ_API_KEY"] = ""
from app.ingest import load_all
from app.rewrite import rewrite
from app.retrieval import Index
from app.answer import sufficient

Q = json.load(open(os.path.join(os.path.dirname(__file__), "queries.json")))
idx = Index(load_all())
svcs = sorted({d["meta"]["service"] for d in idx.docs if d["type"] == "log"})
CONFIGS = {"dense_only": "dense", "bm25_dense_rrf": "rrf", "rewrite_hybrid_rerank": "full"}

def ndcg(rels):
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal)) or 1
    return dcg / idcg if any(rels) else 0.0

rows, report = [], {}
for name, mode in CONFIGS.items():
    rec, mrr, nd, lat, abst_ok, n_ans = [], [], [], [], [], 0
    by_type = {}
    for q in Q:
        rw = rewrite(q["query"], svcs)
        t = time.time(); hits, _ = idx.search(rw, mode=mode, k=10); ms = (time.time() - t) * 1000
        lat.append(ms)
        if q["expect_abstain"]:
            abst_ok.append(not sufficient(hits, rw)); continue
        rels = [1 if h["doc"]["meta"].get("error_id") == q["expected_error_id"] else 0 for h in hits]
        r5 = float(any(rels[:5])); rr = next((1 / (i + 1) for i, r in enumerate(rels) if r), 0.0)
        rec.append(r5); mrr.append(rr); nd.append(ndcg(rels))
        by_type.setdefault(q["type"], []).append(r5)
    lat.sort()
    report[name] = {"recall@5": round(sum(rec) / len(rec), 3), "mrr@10": round(sum(mrr) / len(mrr), 3), "ndcg@10": round(sum(nd) / len(nd), 3),
                    "abstention_accuracy": round(sum(abst_ok) / len(abst_ok), 3),
                    "recall@5_by_type": {k: round(sum(v) / len(v), 3) for k, v in by_type.items()},
                    "latency_ms_p50": round(lat[len(lat) // 2], 1), "latency_ms_p95": round(lat[int(len(lat) * .95) - 1], 1),
                    "llm_cost_usd": 0.0}
print(json.dumps(report, indent=1))
json.dump(report, open(os.path.join(os.path.dirname(__file__), "results.json"), "w"), indent=1)
