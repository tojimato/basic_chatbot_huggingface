"""Minimal Flask chatbot using Hugging Face Transformers.

This module exposes POST /chatbot which accepts JSON {"prompt": "..."}
and returns a generated reply. The code below is intentionally simple and
keeps an in-memory bounded conversation history.
"""

import logging
import os
from collections import deque
import threading
from typing import Optional
from flask import Response
from transformers import TextIteratorStreamer

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "facebook/blenderbot-400M-distill")
HISTORY_MAX_LEN = int(os.environ.get("HISTORY_MAX_LEN", "20"))

# Globals for model and tokenizer (loaded lazily)
model = None
tokenizer = None

# Bounded conversation history to avoid unbounded memory growth
conversation_history = deque(maxlen=HISTORY_MAX_LEN)


def load_model() -> None:
    """Lazily load model and tokenizer into module globals."""
    global model, tokenizer
    if model is None or tokenizer is None:
        logger.info("Loading model %s...", MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info("Model loaded")


def _extract_prompt() -> Optional[str]:
    """Extract `prompt` from the incoming request.

    Accepts JSON body {"prompt": "..."} or form-encoded `prompt`.
    Returns the prompt string or None if not present/invalid.
    """
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data.get("prompt")

    # Accept form-encoded bodies
    if request.form and "prompt" in request.form:
        return request.form.get("prompt")

    # Log raw body for diagnostics and return None
    raw = request.get_data(as_text=True)
    if raw:
        logger.warning("Received invalid payload (not JSON/form): %s", raw[:1000])
    return None


@app.route("/chatbot", methods=["POST"])
def handle_prompt():
    """Handle a chat prompt and return a generated reply as plain text."""
    prompt = _extract_prompt()
    if not prompt:
        return jsonify({"error": "Invalid request: expected JSON with 'prompt' or form field 'prompt'"}), 400

    # Ensure model is loaded
    try:
        load_model()
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        return jsonify({"error": "Model not available"}), 500

    # Prepare history and inputs
    history = "\n".join(conversation_history)
    inputs = tokenizer.encode_plus(history, prompt, return_tensors="pt")

    # Generate reply (ensure beam search is disabled for deterministic single-output)
    outputs = model.generate(**inputs, max_length=60, num_beams=1)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Update history (bounded)
    conversation_history.append(prompt)
    conversation_history.append(reply)

    return reply, 200


@app.route("/chatbot/stream", methods=["POST"])
def stream_prompt():
    """Stream generation tokens as Server-Sent Events (SSE).

    Clients can connect to `/chatbot/stream` with a JSON body `{"prompt":"..."}`
    and receive partial text as events (`data: <chunk>\n\n`).
    """
    prompt = _extract_prompt()
    if not prompt:
        return jsonify({"error": "Invalid request: expected JSON with 'prompt' or form field 'prompt'"}), 400

    try:
        load_model()
    except Exception as e:
        logger.exception("Failed to load model for streaming: %s", e)
        return jsonify({"error": "Model not available"}), 500

    # Prepare inputs
    history = "\n".join(conversation_history)
    inputs = tokenizer.encode_plus(history, prompt, return_tensors="pt")

    # Set up streamer and background generation
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    def gen_thread():
        try:
            # streamer requires num_beams == 1. Force sampling to allow streaming.
            model.generate(**inputs, max_length=60, streamer=streamer, num_beams=1, do_sample=True)
        except Exception:
            # Let streamer finish/raise
            logger.exception("Generation thread failed")

    threading.Thread(target=gen_thread, daemon=True).start()

    # Stream SSE events
    def event_stream():
        full = []
        for chunk in streamer:
            full.append(chunk)
            # SSE frame
            yield f"data: {chunk}\n\n"
        # After generation complete, update history
        reply = "".join(full).strip()
        if reply:
            conversation_history.append(prompt)
            conversation_history.append(reply)

    return Response(event_stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    # Allow overriding host/port via env vars for local testing
    host = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_RUN_PORT", "5000"))
    app.run(host=host, port=port)