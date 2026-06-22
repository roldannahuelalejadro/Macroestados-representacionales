from __future__ import annotations

import json
import urllib.error
import urllib.request


def post(path: str, payload: dict) -> None:
    data_bytes = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:11434{path}",
        data=data_bytes,
        headers={"Content-Type": "application/json"},
    )
    print(f"endpoint {path}")
    try:
        data = json.loads(urllib.request.urlopen(request, timeout=60).read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print("ERR", type(exc).__name__, exc.code, body[:1000])
        return
    except Exception as exc:
        print("ERR", type(exc).__name__, str(exc)[:1000])
        return

    embeddings = data.get("embeddings", [])
    embedding = data.get("embedding")
    print("OK")
    print("keys", sorted(data.keys()))
    print("n", len(embeddings) if embeddings else (1 if embedding else 0))
    print("dim", len(embeddings[0]) if embeddings else (len(embedding) if embedding else None))


post("/api/embed", {"model": "qwen3-coder:30b-32k", "input": ["What is S1_SHORT_CODE?"]})
post("/api/embeddings", {"model": "qwen3-coder:30b-32k", "prompt": "What is S1_SHORT_CODE?"})
