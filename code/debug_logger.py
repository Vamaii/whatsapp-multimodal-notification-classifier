"""
Writes one JSON-lines record per message to debug_trace.jsonl, capturing
every pipeline stage: evidence card -> unified text -> prompt -> raw LLM
response -> validated decision. If msg_048 comes back mute when you
expected notify, this is what tells you whether the evidence builder,
the prompt, the model, or the validator is responsible.
"""
import json
import os

LOG_PATH = os.path.join(os.path.dirname(__file__), "debug_trace.jsonl")

_fh = None


def start_run():
    global _fh
    _fh = open(LOG_PATH, "w", encoding="utf-8")


def log_message(message_id, evidence, unified_text, user_prompt, raw_llm_response,
                 final_result):
    if _fh is None:
        start_run()
    record = {
        "message_id": message_id,
        "evidence_card": evidence,
        "unified_text": unified_text,
        "prompt_sent": user_prompt,
        "raw_llm_response": raw_llm_response,
        "final_decision": final_result,
    }
    _fh.write(json.dumps(record, default=str) + "\n")
    _fh.flush()


def close_run():
    global _fh
    if _fh is not None:
        _fh.close()
        _fh = None


def inspect_one(message_id: str):
    """Pretty-print the full trace for one message_id from the log file."""
    if not os.path.exists(LOG_PATH):
        print(f"No debug log found at {LOG_PATH} -- run main.py first.")
        return
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["message_id"] == message_id:
                _print_trace(rec)
                return
    print(f"{message_id} not found in {LOG_PATH}")


def _print_trace(rec):
    def section(title, obj):
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
        if isinstance(obj, (dict, list)):
            print(json.dumps(obj, indent=2, default=str))
        else:
            print(obj)

    section("MESSAGE ID", rec["message_id"])
    section("UNIFIED TEXT (post media-normalization)", rec["unified_text"])
    section("EVIDENCE CARD", rec["evidence_card"])
    section("PROMPT SENT TO LLM", rec["prompt_sent"])
    section("RAW LLM RESPONSE", rec["raw_llm_response"])
    section("FINAL DECISION (post-validator)", rec["final_decision"])
