"""
Normalizes image and voice-note messages into text using Gemini's
multimodal input, via the shared llm_clients.gemini_understand_media
(so this gets the same retry/backoff/logging as every other LLM call,
instead of a separate, weaker implementation). update gemini not working quota finished so using others. 
"""
import os
import json
import mimetypes
# from llm_clients import gemini_understand_media, openrouter_understand_media
from llm_clients import openrouter_understand_media

PROMPT = (
    "You are looking at a WhatsApp message attachment (image poster/screenshot, "
    "or a transcribed voice note). Extract:\n"
    "1. transcribed_text: all readable text / spoken words, verbatim.\n"
    "2. gist: one short sentence describing what this is and its apparent intent "
    "(e.g. 'promotional poster for a 50% sale', 'voice note urgently asking for money').\n"
    "3. urgency_signal: true if it conveys time pressure or urgency, else false.\n"
    "Respond ONLY as compact JSON: "
    '{"transcribed_text": "...", "gist": "...", "urgency_signal": true|false}'
)


def understand_media(file_path: str, dataset_dir: str) -> dict:
    """
    file_path: relative path from dataset.csv, e.g. 'media/images/img_001.jpg'
    Returns dict with transcribed_text / gist / urgency_signal.
    Falls back to a safe empty result on any API error (after retries via
    llm_clients) so the pipeline keeps running -- log the error, don't
    crash the batch.
    """
    full_path = os.path.join(dataset_dir, file_path)
    if not os.path.exists(full_path):
        return {
            "transcribed_text": "",
            "gist": f"[file not found: {file_path}]",
            "urgency_signal": False,
        }

    mime_type, _ = mimetypes.guess_type(full_path)
    if mime_type is None:
        mime_type = "image/jpeg" if file_path.endswith((".jpg", ".jpeg", ".png")) else "audio/mpeg"

    try:
        with open(full_path, "rb") as fh:
            file_bytes = fh.read()
        text = openrouter_understand_media(PROMPT, file_bytes, mime_type)
        return json.loads(text)
    except Exception as e:
        return {
            "transcribed_text": "",
            "gist": f"[media understanding failed: {e}]",
            "urgency_signal": False,
        }


if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    result = understand_media("media/images/img_001.jpg", dataset_dir)
    print(result)
