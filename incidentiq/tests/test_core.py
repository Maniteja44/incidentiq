import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["GROQ_API_KEY"] = ""   # tests must not call the LLM
import pytest
from fastapi.testclient import TestClient
from app.rewrite import rewrite
from app.retrieval import rrf, tokenize
from app.main import app

c = TestClient(app)

def test_rewrite_preserves_error_ids_and_filters():
    rw = rewrite("POSTGRES_DEADLOCK in orders-api production trace 7cf07bb3f343d9eb")
    assert "POSTGRES_DEADLOCK" in rw["error_ids"] and "POSTGRES_DEADLOCK" in rw["rewritten"]
    assert rw["filters"]["environment"] == "production" and rw["filters"]["trace_id"] == "7cf07bb3f343d9eb"

def test_rrf_orders_by_agreement():
    order, _ = rrf([[1, 2, 3], [3, 1, 4]])
    assert order[0] == 1 and set(order) == {1, 2, 3, 4}

def test_tokenizer_keeps_codes():
    assert "k8s_crashloop" in tokenize("K8S_CRASHLOOP") and "crashloop" in tokenize("K8S_CRASHLOOP")

def test_health():
    assert c.get("/health").json()["status"] == "ok"

def test_search_exact_code():
    j = c.post("/search", json={"query": "REDIS_OOM", "mode": "full"}).json()
    assert any(h["meta"].get("error_id") == "REDIS_OOM" for h in j["hits"][:3])

def test_answer_has_citations_and_location():
    j = c.post("/answer", json={"query": "QueuePool limit of size 5 overflow 10 reached"}).json()
    assert not j["abstained"] and j["citations"] and j["occurrences"]["error_id"] == "SQLALCHEMY_POOL_TIMEOUT"

def test_abstains_on_unrelated():
    j = c.post("/answer", json={"query": "how do I bake sourdough bread"}).json()
    assert j["abstained"] and j["citations"] == []

def test_ingest_requires_token():
    assert c.post("/ingest").status_code == 401
    assert c.post("/ingest", headers={"x-admin-token": "dev-admin-token"}).status_code == 200
