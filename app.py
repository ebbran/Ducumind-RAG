import json
import threading
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.utils import secure_filename

import config
from rag.ingest import build_index
from rag.pipeline import RAGPipeline

app = Flask(__name__)
_pipe, _lock = None, threading.Lock()


def pipe():
    global _pipe
    with _lock:
        if _pipe is None:
            _pipe = RAGPipeline()
        return _pipe


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    p = pipe()
    return jsonify({"ready": p.ready, "chunks": len(p.index.chunks), "llm": config.LLM_PROVIDER})


@app.post("/api/upload")
def upload():
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for f in request.files.getlist("files"):
        name = secure_filename(f.filename)
        if name.lower().endswith((".pdf", ".txt", ".md")):
            f.save(config.DOCS_DIR / name)
    n = build_index()
    pipe().reload()
    return jsonify({"chunks": n})


@app.post("/api/chat")
def chat():
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "empty question"}), 400

    def gen():
        try:
            for ev in pipe().ask_stream(question, body.get("history", []), body.get("expand", True)):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(stream_with_context(gen()), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
