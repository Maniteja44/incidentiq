import os, httpx
URL = "https://api.groq.com/openai/v1/chat/completions"

def groq_chat(messages, temperature=0.1, max_tokens=700):
    key = os.getenv("GROQ_API_KEY")
    if not key: raise RuntimeError("GROQ_API_KEY not set")
    r = httpx.post(URL, headers={"Authorization": f"Bearer {key}"}, timeout=30,
                   json={"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                         "messages": messages, "temperature": temperature, "max_tokens": max_tokens})
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"], j.get("usage", {})
