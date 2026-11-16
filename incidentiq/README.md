# IncidentIQ
RAG chatbot over `data/logs_1000.jsonl` + `data/error_knowledge.csv`. Read-only guidance; never executes commands.

Pipeline: query rewrite (keeps error codes/trace ids, extracts service/env filters) -> BM25 + dense -> RRF -> rerank -> citation-grounded Groq answer, with abstention when evidence is weak.

## Run
    python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
    cp .env.example .env   # put your GROQ_API_KEY in .env
    uvicorn app.main:app --reload     # UI at http://localhost:8000
    pytest -q tests

Set `EMBED_BACKEND=st` (and `pip install sentence-transformers`) for bge-small embeddings + BGE cross-encoder; default `lite` needs no downloads.

Endpoints: GET /health, POST /search, POST /answer, POST /ingest (header `x-admin-token`), POST /feedback.

## Evaluate
    python eval/run_eval.py     # 40 queries (10 exact, 10 paraphrase, 10 multi-clue, 10 unanswerable) x 3 configs -> eval/results.json
Relevance is judged at error_id level. Retrieval-only, so LLM cost is 0. Queries were written by hand from the knowledge base; they are not a truly held-out set.

## Deploy
- Docker: `docker compose up --build`
- Render: connect the repo (uses `render.yaml`), set `GROQ_API_KEY`, then set `CORS_ORIGINS` to your frontend URL.
- Frontend: paste the prompt from the chat into Lovable and set the backend URL in its Settings.
