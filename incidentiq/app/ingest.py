"""Ingestion: knowledge entries + log events -> chunks with metadata."""
import csv, json, os
DATA = os.path.join(os.path.dirname(__file__), "..", "data")

def load_knowledge(path=None):
    path = path or os.path.join(DATA, "error_knowledge.csv")
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            text = (f"{r['error_id']} ({r['technology']}): {r['error_example']}. "
                    f"Possible causes: {r['possible_causes']}. "
                    f"Diagnostic steps: {r['general_diagnostic_steps']}.")
            out.append({"id": f"kb:{r['error_id']}", "type": "knowledge", "text": text,
                        "meta": {"error_id": r["error_id"], "technology": r["technology"],
                                 "source_title": r["source_title"], "source_url": r["source_url"],
                                 "source_status": r["source_status"]}})
    return out

def load_logs(path=None):
    path = path or os.path.join(DATA, "logs_1000.jsonl")
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (f"{r.get('timestamp')} {r.get('level')} service={r.get('service')} "
                    f"env={r.get('environment')} trace_id={r.get('trace_id')} "
                    f"error_id={r.get('error_id')} {r.get('message')}")
            out.append({"id": f"log:{i}", "type": "log", "text": text,
                        "meta": {"line": i, "timestamp": r.get("timestamp"), "level": r.get("level"),
                                 "service": r.get("service"), "environment": r.get("environment"),
                                 "trace_id": r.get("trace_id"), "error_id": r.get("error_id"),
                                 "technology": r.get("technology"), "synthetic": r.get("synthetic", False),
                                 "message": r.get("message")}})
    return out

def load_all():
    return load_knowledge() + load_logs()
