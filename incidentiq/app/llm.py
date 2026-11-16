import os, httpx
URL = "https://api.groq.com/openai/v1/chat/completions"
FALLBACK_MODELS = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]

def _models():
    first = os.getenv("GROQ_MODEL")
    return ([first] if first else []) + [m for m in FALLBACK_MODELS if m != first]

def groq_chat(messages, temperature=0.1, max_tokens=700):
    key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    if not key: raise RuntimeError("GROQ_API_KEY is not set on the server")
    errors = []
    for model in _models():
        try:
            r = httpx.post(URL, headers={"Authorization": f"Bearer {key}"}, timeout=40,
                           json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        except httpx.HTTPError as e:
            errors.append(f"{model}: network {type(e).__name__}"); continue
        if r.status_code in (401, 403):
            raise RuntimeError(f"Groq HTTP {r.status_code}: API key rejected (check GROQ_API_KEY)")
        if r.status_code >= 400:
            try: msg = r.json().get("error", {}).get("message", "")[:150]
            except Exception: msg = r.text[:150]
            errors.append(f"{model}: HTTP {r.status_code} {msg}"); continue
        j = r.json()
        text = (j["choices"][0]["message"].get("content") or "").strip()
        if text:
            return text, {**j.get("usage", {}), "model": model}
        errors.append(f"{model}: empty reply")
    raise RuntimeError(" | ".join(errors)[:400])
