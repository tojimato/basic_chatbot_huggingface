#!/usr/bin/env python3
"""Simple tester for the chatbot endpoints.

Usage:
  python tests/stream_test.py --stream    # test /chatbot/stream (SSE)
  python tests/stream_test.py            # test /chatbot (blocking)
"""
import argparse
import sys
import requests

BASE = "http://127.0.0.1:5000"


def test_blocking(prompt: str):
    resp = requests.post(f"{BASE}/chatbot", json={"prompt": prompt})
    resp.raise_for_status()
    print("REPLY:")
    print(resp.text)


def test_stream(prompt: str):
    with requests.post(f"{BASE}/chatbot/stream", json={"prompt": prompt}, stream=True) as r:
        r.raise_for_status()
        print("Streaming reply (SSE frames):")
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            # SSE frames are like: data: <chunk>
            if line.startswith("data:"):
                chunk = line.split("data:", 1)[1].strip()
                print(chunk, end="", flush=True)
            else:
                # other SSE metadata
                print(f"\n[{line}]\n", end="", flush=True)
        print()  # newline at end


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stream", action="store_true", help="Use streaming endpoint (/chatbot/stream)")
    p.add_argument("--prompt", default="Hello, how are you today?", help="Prompt to send")
    args = p.parse_args()

    try:
        if args.stream:
            test_stream(args.prompt)
        else:
            test_blocking(args.prompt)
    except requests.HTTPError as e:
        print("HTTP error:", e, file=sys.stderr)
    except Exception as e:
        print("Error:", e, file=sys.stderr)


if __name__ == "__main__":
    main()
