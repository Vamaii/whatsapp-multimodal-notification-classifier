"""
Small, deliberately dumb sanity-check layer. Doesn't make the system
smarter js hoping it  catches cases where the LLM's own output is internally
inconsistent or violates the output contract, before it reaches
output.csv.
"""
import re

ALLOWED_ACTIONS = {"notify", "digest", "mute"}
ALLOWED_TYPES = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
}

# action -> reason must NOT strongly imply the opposite urgency
MUTE_CONTRADICTION_WORDS = re.compile(
    r"\b(urgent|emergency|immediately|life[- ]threatening|critical)\b", re.I
)
NOTIFY_LOW_VALUE_WORDS = re.compile(
    r"\b(low[- ]value|repetitive|routine spam|not important)\b", re.I
)


def validate_and_fix(result: dict, ev: dict) -> dict:
    # ---- schema enforcement ----
    if result["action"] not in ALLOWED_ACTIONS:
        result["action"] = "digest"
        result["reason"] = (result.get("reason") or "") + " [validator: invalid action, defaulted to digest]"
    if result["message_type"] not in ALLOWED_TYPES:
        result["message_type"] = "unknown"

    try:
        conf = float(result["confidence"])
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    # ---- action/reason contradiction check ----
    reason = result.get("reason") or ""
    if result["action"] == "mute" and MUTE_CONTRADICTION_WORDS.search(reason):
        conf = min(conf, 0.4)
        reason += " [validator: reason language conflicts with mute action -- confidence reduced]"
    if result["action"] == "notify" and NOTIFY_LOW_VALUE_WORDS.search(reason):
        conf = min(conf, 0.4)
        reason += " [validator: reason language conflicts with notify action -- confidence reduced]"

    # ---- message_type needs at least plausible support ----
    if result["message_type"] == "scam" and not (ev.get("flag_otp_request") or ev.get("flag_payment_link") or ev.get("business_domain_mismatch")):
        # LLM called it a scam without any of the code-detectable risk signals present --
        # not necessarily wrong (content-only scams exist), but confidence shouldn't be maxed
        conf = min(conf, 0.75)

    # ---- confidence must reflect evidence thinness ----
    quality = ev.get("evidence_quality", 0.0)
    if not ev.get("has_history", False) and conf > 0.75:
        conf = 0.75
        reason += " [validator: no history for this source -- confidence capped]"
    elif quality < 0.3 and conf > 0.85:
        conf = 0.85
        reason += " [validator: thin evidence -- confidence capped]"

    # ---- confidence must reflect the LLM's OWN ledger agreeing with itself ----
    # Models tend to be overconfident; a stated confidence of 0.9+ means
    # nothing if the model's own risk/consequence scores don't support the
    # action it picked. Don't trust the number -- check the reasoning.
    ledger = result.get("_ledger") or {}

    def _score(dim):
        try:
            return float(ledger.get(dim, {}).get("score"))
        except (TypeError, ValueError, AttributeError):
            return None

    risk = _score("risk")
    consequence = _score("consequence_of_ignoring")
    urgency = _score("urgency")

    if risk is not None and risk >= 7 and result["action"] == "notify" and conf > 0.6:
        # high self-reported risk but still chose to notify -- internally
        # inconsistent unless explicitly justified; don't let confidence ride high
        conf = min(conf, 0.6)
        reason += " [validator: model's own risk score is high but action is notify -- confidence capped]"
    if (
        consequence is not None and urgency is not None
        and consequence <= 2 and urgency <= 2
        and result["action"] == "notify" and conf > 0.7
    ):
        conf = min(conf, 0.7)
        reason += " [validator: low urgency/consequence scores but action is notify -- confidence capped]"
    if result["action"] == "mute" and consequence is not None and consequence >= 8 and conf > 0.5:
        conf = min(conf, 0.5)
        reason += " [validator: model's own consequence score is high but action is mute -- confidence capped]"

    result["confidence"] = round(conf, 3)
    result["reason"] = reason.strip()

    # ---- evidence_message_ids must be real ids or "none" ----
    ids = result.get("evidence_message_ids", "none")
    if not ids or ids == "":
        ids = "none"
    result["evidence_message_ids"] = ids

    return result
