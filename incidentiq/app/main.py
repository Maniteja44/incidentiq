import os, time, json
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .ingest import load_all
from .rewrite import rewrite
from .retrieval import Index
from .answer import answer

app = FastAPI(title="IncidentIQ")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","), allow_methods=["*"], allow_headers=["*"])
STATE = {"index": None}
FEEDBACK = os.path.join(os.path.dirname(__file__), "..", "data", "feedback.jsonl")

def idx():
    if STATE["index"] is None: STATE["index"] = Index(load_all())
    return STATE["index"]

class Q(BaseModel):
    query: str
    mode: str = "full"   # dense | bm25 | rrf | full
    k: int = 5
class FB(BaseModel):
    query: str
    helpful: bool
    comment: str = ""

def _search(q: Q):
    i = idx()
    svcs = sorted({d["meta"]["service"] for d in i.docs if d["type"] == "log"})
    rw = rewrite(q.query, svcs)
    mode = {"rrf": "rrf"}.get(q.mode, q.mode)
    t = time.time()
    hits, trace = i.search(rw, mode=mode, k=q.k)
    return i, rw, hits, trace, int((time.time() - t) * 1000)

@app.get("/health")
def health():
    n = len(STATE["index"].docs) if STATE["index"] else 0
    return {"status": "ok", "indexed": n, "llm_configured": bool(os.getenv("GROQ_API_KEY"))}

@app.post("/search")
def search(q: Q):
    i, rw, hits, trace, ms = _search(q)
    return {"rewrite": rw, "trace": trace, "latency_ms": ms,
            "hits": [{"id": h["doc"]["id"], "type": h["doc"]["type"], "score": h["score"], "bm25": h["bm25"], "dense": h["dense"], "text": h["doc"]["text"], "meta": h["doc"]["meta"]} for h in hits]}

@app.post("/answer")
def answer_ep(q: Q):
    i, rw, hits, trace, ms = _search(q)
    out = answer(i, rw, hits, trace)
    out.update({"rewrite": rw, "trace": trace, "retrieval_ms": ms,
                "hits": [{"id": h["doc"]["id"], "type": h["doc"]["type"], "score": h["score"]} for h in hits]})
    return out

@app.post("/ingest")
def ingest(x_admin_token: str = Header(None)):
    if x_admin_token != os.getenv("ADMIN_TOKEN", "dev-admin-token"): raise HTTPException(401, "invalid admin token")
    STATE["index"] = Index(load_all())
    return {"indexed": len(STATE["index"].docs)}

@app.post("/feedback")
def feedback(f: FB):
    with open(FEEDBACK, "a") as fh: fh.write(json.dumps(f.model_dump()) + "\n")
    return {"ok": True}

@app.get("/")
def ui(): return FileResponse(os.path.join(os.path.dirname(__file__), "..", "static", "index.html"))
