"""Query rewrite: extracts filters/identifiers; exact error codes are preserved verbatim."""
import re
ERR_ID = re.compile(r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b")
TRACE = re.compile(r"\b[0-9a-f]{16}\b")
HTTP = re.compile(r"\b(?:HTTP/1\.1\s+)?([45]\d\d)\b")
SQLSTATE = re.compile(r"\b\d{2}[0-9A-Z]{3}\b(?=\))")
ERRNO = re.compile(r"Errno\s*-?\d+", re.I)
EXC = re.compile(r"\b\w+(?:\.\w+)*(?:Error|Exception|Timeout)\b")
SERVICES = ["auth-api", "orders-api", "payments-api", "billing-api", "gateway", "worker", "search-api",
            "notifications", "inventory-api", "user-service"]
SYN = {"db": "database postgres", "oom": "OOM out of memory", "memory": "memory OOM",
       "cant connect": "connection refused", "can't connect": "connection refused",
       "slow": "timeout latency", "full disk": "no space left on device",
       "certificate": "certificate expired TLS", "rate limit": "429 too many requests"}

def rewrite(q, known_services=()):
    ids = ERR_ID.findall(q)
    traces = TRACE.findall(q)
    codes = list({*HTTP.findall(q), *ERRNO.findall(q), *EXC.findall(q), *SQLSTATE.findall(q)})
    filters = {}
    ql = q.lower()
    for s in list(known_services) + SERVICES:
        if s.lower() in ql: filters["service"] = s; break
    for e, v in (("production", "production"), ("prod", "production"), ("staging", "staging"), ("dev", "dev")):
        if re.search(rf"\b{e}\b", ql): filters["environment"] = v; break
    for lv in ("CRITICAL", "ERROR", "WARNING", "WARN"):
        if re.search(rf"\b{lv}\b", q): filters["level"] = lv; break
    if traces: filters["trace_id"] = traces[0]
    exp = [v for k, v in SYN.items() if k in ql]
    rewritten = " ".join([q] + exp)
    return {"original": q, "rewritten": rewritten, "preserved": ids + codes + traces, "error_ids": ids, "filters": filters}
