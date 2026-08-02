"""
Decision engine.
making sure that the causalities and false positives are minimized and the decision is made based on the evidence and not on the assumptions.
Order matters:
1. Hard safety overrides (OTP/payment/domain-mismatch pattern) run in
   plain Python, BEFORE any LLM call. If triggered, the LLM is never
   consulted for that message -- the override is final. This matches
   the spec's explicit instruction that risk should override usual
   engagement, and it saves an API call.
2. Otherwise, call Groq with the evidence card and parse its structured
   response.
3. Run the response through the validator before accepting it.
"""
import json
import sys
from reasoners import get_reasoner
from prompt_builder import SYSTEM_PROMPT, build_user_prompt
from validator import validate_and_fix

_reasoner = get_reasoner()


def _hard_override(ev: dict):
    if ev.get("hard_override_scam"):
        return {
            "message_id": ev["message_id"],
            "action": "mute",
            "message_type": "scam",
            "reason": (
                "Hard safety override: message either directly requests the user "
                "share/enter an OTP, or comes from a sender domain that doesn't "
                "match the official business domain combined with account-risk "
                "language (verify/restricted/wallet/refund/PIN, etc). Muted "
                "regardless of the user's usual engagement, per safety policy."
            ),
            "confidence": 0.97,
            "evidence_message_ids": "none",
            "_source": "hard_override",
        }
    return None


def _safe_fallback(ev: dict, error: str) -> dict:
    """
    Used when the reasoning call fails after retries, or returns
    unparseable output. Defaults to digest (not mute, not notify) --
    the least harmful wrong guess -- with low confidence and a reason
    that makes the failure visible in output.csv rather than silently
    guessing.
    """
    return {
        "message_id": ev["message_id"],
        "action": "digest",
        "message_type": "unknown",
        "reason": f"LLM call failed after retries, defaulted to digest for manual review: {error}",
        "confidence": 0.1,
        "evidence_message_ids": "none",
        "_source": "fallback",
    }


def decide(unified_text: str, ev: dict) -> dict:
    override = _hard_override(ev)
    if override is not None:
        return override

    user_prompt = build_user_prompt(unified_text, ev)

    try:
        raw = _reasoner.decide_raw(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        print(f"[decision] {ev['message_id']}: reasoning call failed -- {e}", file=sys.stderr)
        return _safe_fallback(ev, str(e))

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[decision] {ev['message_id']}: could not parse LLM JSON -- {e}", file=sys.stderr)
            return _safe_fallback(ev, f"unparseable response: {raw[:200]}")

    result = {
        "message_id": ev["message_id"],
        "action": parsed.get("action", "digest"),
        "message_type": parsed.get("message_type", "unknown"),
        "reason": parsed.get("reason", ""),
        "confidence": parsed.get("confidence", 0.5),
        "evidence_message_ids": parsed.get("evidence_message_ids") or "none",
        "_ledger": parsed.get("evidence_ledger", {}),
        "_source": "llm",
        "_raw_llm_response": raw,
    }
    if isinstance(result["evidence_message_ids"], list):
        result["evidence_message_ids"] = (
            ";".join(result["evidence_message_ids"]) if result["evidence_message_ids"] else "none"
        )

    return validate_and_fix(result, ev)
