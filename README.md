# Basic Chatbot

A minimal Flask + Transformers chatbot demo that exposes a single `/chatbot` POST endpoint. The app uses a pretrained seq2seq model (BlenderBot) from Hugging Face Transformers.

## Features
- Simple REST endpoint for chat: POST `/chatbot`
- Uses `transformers` for model/tokenizer loading and generation

## Requirements
- Python 3.8+ (recommended)
- See `requirements.txt` for pinned dependencies

## Setup (Windows)
1. Activate the provided virtual environment (if using it):

```powershell
myenv\Scripts\Activate.ps1
# or for cmd: myenv\Scripts\activate.bat
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run
Start the Flask app:

```powershell
python src/app.py
```

The server will listen on `http://127.0.0.1:5000` by default.

## Usage
Send a POST request with JSON body containing `prompt` to `/chatbot`.

Example curl:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Hello, how are you?"}' http://127.0.0.1:5000/chatbot
```

Example Python snippet (cross-platform):

```python
import requests

# Simple, cross-platform example (works on Windows, macOS, Linux)
resp = requests.post(
	'http://127.0.0.1:5000/chatbot',
	json={'prompt': 'Hello, how are you today?'}
)
print(resp.text)
```

## Testing
Quick commands to test the blocking and streaming endpoints (from project root).

- Start the server:

```powershell
python src/app.py
```

- Blocking test (returns full reply):

```powershell
python tests/stream_test.py
```

- Streaming test (Server-Sent Events):

```powershell
python tests/stream_test.py --stream
```

- Alternative: use `curl` (use `curl.exe` on PowerShell to avoid aliasing):

```powershell
curl.exe -N -H "Accept: text/event-stream" -H "Content-Type: application/json" -d "{\"prompt\":\"Hello, how are you today?\"}" http://127.0.0.1:5000/chatbot/stream
```

- Browser client example (JavaScript EventSource):

```html
<script>
	const evtSrc = new EventSource('/chatbot/stream', { withCredentials: false });
	evtSrc.onmessage = (e) => {
		// e.data contains partial chunks as they arrive
		console.log('chunk', e.data);
	};
</script>
```

Notes:
- Use the Python test script when `curl` buffering or PowerShell quoting causes issues; it works consistently across platforms.
- Streaming uses SSE frames (`data: <chunk>\n\n`); web browsers can consume via `EventSource` and curl with `-N`.

## Notes
- The model downloads weights on first run and may be large; ensure you have enough disk space and a good network connection.
- Generation currently appends conversation history in-memory; this may grow unbounded for a long-running process.

## Project Structure

- `src/app.py` — Flask app and `/chatbot` endpoint
- `src/chatbot.py`, `src/main.py` — additional helpers (if present)
- `requirements.txt` — Python dependencies

## License
This repository contains example/demo code — adjust licensing as needed.
