"""
Env vars required for real runs:
    GEMINI_API_KEY   -- media understanding (images, voice notes)
    # XAI_API_KEY      -- reasoning/decision calls (Groq, default reasoner)
    REASONER         -- "groq" (default) or "gemini", for A/B testing
    GROQ_API_KEY     -- API key for the Groq reasoning service
"""
import sys
import csv
import os
import argparse
from dotenv import load_dotenv
load_dotenv()
from loader import Dataset
from evidence import build_evidence
from decision import decide, _hard_override
from media import understand_media
import debug_logger

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
OUTPUT_PATH = os.path.join(DATASET_DIR, "output.csv")

OUTPUT_COLUMNS = [
    "message_id", "action", "message_type", "reason", "confidence",
    "evidence_message_ids",
]


def get_unified_text(ds, msg_row, media_cache, dry_run):
    media_type = msg_row.media_type if isinstance(msg_row.media_type, str) else ""
    media_id = msg_row.media_id if isinstance(msg_row.media_id, str) else ""

    if media_type not in ("image", "voice") or not media_id:
        return msg_row.message_text if isinstance(msg_row.message_text, str) else ""

    if media_id in media_cache:
        return media_cache[media_id]

    if dry_run:
        text = "[dry-run placeholder for %s %s]" % (media_type, media_id)
        media_cache[media_id] = text
        return text

    path = ds.image_path.get(media_id) if media_type == "image" else ds.voice_path.get(media_id)
    if not path:
        text = "[%s message, file not found: %s]" % (media_type, media_id)
        media_cache[media_id] = text
        return text

    try:
        result = understand_media(path, DATASET_DIR)
        text = ("%s (gist: %s)" % (result.get('transcribed_text', ''), result.get('gist', ''))).strip()
    except Exception as e:
        print("[main] media understanding failed for %s (%s): %s" % (media_id, path, e), file=sys.stderr)
        text = "[%s message, media understanding failed: %s]" % (media_type, media_id)
    media_cache[media_id] = text
    return text


def run(dry_run, limit):
    ds = Dataset()
    media_cache = {}
    rows_out = []
    debug_logger.start_run()

    messages = list(ds.messages.itertuples())
    if limit:
        messages = messages[:limit]

    total = len(messages)
    for i, msg_row in enumerate(messages, 1):
        ev = build_evidence(ds, msg_row)
        unified_text = get_unified_text(ds, msg_row, media_cache, dry_run)

        if dry_run:
            override = _hard_override(ev)
            if override is not None:
                result = override
            else:
                result = {
                    "message_id": ev["message_id"],
                    "action": "digest",
                    "message_type": "unknown",
                    "reason": "[dry-run stub -- no LLM called]",
                    "confidence": 0.5,
                    "evidence_message_ids": ";".join(ev["evidence_message_ids"]) or "none",
                }
            raw_response = None
        else:
            result = decide(unified_text, ev)
            raw_response = result.get("_raw_llm_response")

        debug_logger.log_message(
            ev["message_id"], ev, unified_text,
            None if dry_run else "see prompt_builder.build_user_prompt(unified_text, ev)",
            raw_response, result,
        )

        rows_out.append({k: result.get(k, "") for k in OUTPUT_COLUMNS})

        if i % 10 == 0 or i == total:
            print("  processed %d/%d" % (i, total), file=sys.stderr)

    debug_logger.close_run()

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_out)

    print("Wrote %d rows to %s" % (len(rows_out), OUTPUT_PATH))

    missing = set(ds.messages["message_id"]) - {r["message_id"] for r in rows_out}
    if missing:
        print("WARNING: %d message_ids missing from output: %s..." % (len(missing), sorted(missing)[:5]),
              file=sys.stderr)
    else:
        print("OK: one output row for every message_id in messages.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--inspect", type=str, default=None, metavar="MSG_ID")
    args = parser.parse_args()

    if args.inspect:
        debug_logger.inspect_one(args.inspect)
    else:
        run(dry_run=args.dry_run, limit=args.limit)
