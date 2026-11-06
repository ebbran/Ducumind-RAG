"""LLM abstraction: swap Ollama (local/free) and any OpenAI-compatible API with one env var."""
import json
import requests
import config


def stream_chat(messages, temperature=0.2):
    if config.LLM_PROVIDER == "ollama":
        r = requests.post(
            f"{config.OLLAMA_URL}/api/chat",
            json={"model": config.OLLAMA_MODEL, "messages": messages, "stream": True,
                  "options": {"temperature": temperature}},
            stream=True, timeout=180,
        )
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            tok = data.get("message", {}).get("content", "")
            if tok:
                yield tok
            if data.get("done"):
                break
    else:
        r = requests.post(
            f"{config.OPENAI_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
            json={"model": config.OPENAI_MODEL, "messages": messages,
                  "temperature": temperature, "stream": True},
            stream=True, timeout=180,
        )
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0]["delta"].get("content")
            if delta:
                yield delta


def chat(messages, temperature=0.0):
    return "".join(stream_chat(messages, temperature))
