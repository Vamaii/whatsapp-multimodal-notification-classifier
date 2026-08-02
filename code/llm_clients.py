
import os
import sys
import time
import base64
import requests

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2
REQUEST_TIMEOUT = 60


def _log_sizes(label, prompt_text, response_text):
    print(
        f"[llm_clients] {label} | prompt_chars={len(prompt_text)} "
        f"resp_chars={len(response_text)}",
        file=sys.stderr,
    )


def _with_retries(fn, label):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except (requests.RequestException, KeyError, IndexError) as e:
            last_err = e
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(
                f"[llm_clients] {label} attempt {attempt}/{MAX_RETRIES} failed "
                f"({e}), retrying in {wait}s...",
                file=sys.stderr,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)
    raise RuntimeError(f"{label} failed after {MAX_RETRIES} attempts: {last_err}")


def openrouter_understand_media(prompt: str, file_bytes: bytes, mime_type: str) -> str:
    """Single multimodal call: image or audio bytes + instruction -> text."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")

    url = (
        "https://openrouter.ai/api/v1/chat/completions"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://hackerrank.com",
        "X-Title": "Hackerrank Orchestrate"
    }
    if mime_type.startswith("image/"):
        media = {
             "type": "image_url",
             "image_url": {
                 "url": f"data:{mime_type};base64,"
                   f"{base64.b64encode(file_bytes).decode()}"
             },
        }
    else:
        media={
            "type": "input_audio",
             "input_audio": {
                  "data": base64.b64encode(file_bytes).decode(),
                  "format": mime_type.split("/")[-1]
             },
        }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    media
                ]
            }
        ],
        "temperature": 0,
    }

    def _call():
        resp = requests.post(url,headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        # resp.raise_for_status()
        # data = resp.json()
        if not resp.ok:
            print(resp.text)
            resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        _log_sizes("openrouter__media", prompt, text)
        return text

    return _with_retries(_call, "openrouter__media")


def gemini_text(system_prompt: str, user_prompt: str) -> str:
    """Text-only Gemini call, for use as an alternate reasoner (no media bytes)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }

    def _call():
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        if not resp.ok:
            print("Status:", resp.status_code)
            print("Response:", resp.text)
            resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        _log_sizes("gemini_text", system_prompt + user_prompt, text)
        return text

    return _with_retries(_call, "gemini_text")


def groq_reason(system_prompt: str, user_prompt: str) -> str:
    """One reasoning call. Returns raw text (expected to be JSON, caller parses)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,  # deterministic per the project contract
        "response_format": {"type": "json_object"},
    }

    def _call():
        resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        if not resp.ok:
            print("Status:", resp.status_code)
            print("Response:", resp.text)
            resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        _log_sizes("groq_reason", system_prompt + user_prompt, text)
        return text

    return _with_retries(_call, "groq_reason")
