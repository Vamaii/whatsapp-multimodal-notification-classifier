"""
Builds an "evidence card" per message: objective, computable facts only.
No trust scores, no arbitrary weights. Anything requiring judgment
(urgency, novelty, relationship strength, risk) is left to the LLM,
which sees these facts and reasons over them explicitly. Its js saying that everythings is caluclated and not hardcoded.
"""
import re
import math
from datetime import datetime

# Matches a REQUEST to share/enter an OTP -- not just the word "otp" appearing
# (a legitimate delivery notice can say "no OTP is required"; that must NOT match).
OTP_REQUEST_PATTERN = re.compile(
    r"(share|send|enter|type|tell|give|confirm|batao|bhejo)\W{0,20}(the\W{0,5})?"
    r"(otp|one[- ]time password|verification code|code)\b"
    r"|\botp\b\W{0,20}(here|now|share|send|abhi|batao)"
    r"|\bcode\b\W{0,15}(here|now)",
    re.I,
)
OTP_NEGATION_PATTERN = re.compile(
    r"\bno\b\W{0,10}(payment\W{0,5})?\botp\b|\botp\b\W{0,10}(is\W{0,5})?not\W{0,5}required",
    re.I,
)
# Bare or full-URL domains combined with account/payment-risk keywords nearby --
# most of these phishing links in this dataset are bare domains (no http://).
ACCOUNT_RISK_LANGUAGE = re.compile(
    r"\b(verify your (account|profile|identity)|complete verification|"
    r"verification (at|before|link)|account (restricted|suspended|blocked)|"
    r"access (is |will be )?(restricted|blocked|suspended)|"
    r"security (update|alert|risk) (pending|on your)|"
    r"reactivate your|wallet (details|access)|release the (amount|payout)|"
    r"refund (could not|failed|pending).{0,20}(link|click)|"
    r"final verification|\bpin\b.{0,10}(share|enter|confirm)|card access)\b",
    re.I,
)
URGENCY_WORDS = re.compile(
    r"\b(urgent|immediately|act now|last chance|expires? (today|soon)|"
    r"final notice|account (suspended|blocked)|limited time|"
    r"jaldi|abhi|turant|warna|band ho jayega|lock(ed)? ho jayega)\b",
    re.I,
)


def _parse_dnd(window, now: datetime) -> bool:
    if not isinstance(window, str) or "-" not in window:
        return False
    start_s, end_s = window.split("-")
    start = datetime.strptime(start_s.strip(), "%H:%M").time()
    end = datetime.strptime(end_s.strip(), "%H:%M").time()
    t = now.time()
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end


def _evidence_quality(n_history: int) -> float:
    """
    How much historical data exists for this source -- NOT how trustworthy
    it is. 0 history -> 0.0. Grows with log(count), capped at 1.0 by n=10.
    Purely a volume signal so the LLM knows whether it's reasoning from
    rich history or almost none.
    """
    if n_history <= 0:
        return 0.0
    return round(min(1.0, math.log1p(n_history) / math.log1p(10)), 3)


def _event_severity(ev) -> int:
    """Rank how informative a historical reaction is. Report > dismiss >
    reply/open > nothing. Used to pick the 'most important' evidence item."""
    if ev is None:
        return 0
    if ev.message_reported:
        return 4
    if ev.notification_dismissed:
        return 3
    if ev.message_replied:
        return 2
    if ev.message_opened:
        return 1
    return 0


def _tokenize(text):
    if not isinstance(text, str):
        text = ""
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def _pick_diversified_evidence(ds, candidates, current_text, current_flags):
    """
    From a list of historical message rows (already filtered to the
    relevant source), pick up to 3: most recent, most similar (token
    overlap + shared risk flags), most informative reaction (highest
    event severity). Falls back gracefully if the list is short.
    """
    if not candidates:
        return []

    most_recent = candidates[0]  # already sorted desc by created_at in loader

    cur_tokens = _tokenize(current_text)

    def sim_score(row):
        toks = _tokenize(row.message_text)
        overlap = len(cur_tokens & toks)
        flag_match = 0
        row_text = row.message_text if isinstance(row.message_text, str) else ""
        if current_flags["otp"] and OTP_REQUEST_PATTERN.search(row_text):
            flag_match += 1
        if current_flags["urgency"] and URGENCY_WORDS.search(row_text):
            flag_match += 1
        return overlap + 2 * flag_match

    most_similar = max(candidates, key=sim_score)

    most_important = max(
        candidates, key=lambda r: _event_severity(ds.event_by_message.get(r.message_id))
    )

    picked = []
    seen = set()
    for row in (most_recent, most_similar, most_important):
        if row.message_id not in seen:
            picked.append(row)
            seen.add(row.message_id)
    return picked[:3]


def build_evidence(ds, msg_row) -> dict:
    user_id = msg_row.user_id
    text = msg_row.message_text if isinstance(msg_row.message_text, str) else ""
    now = datetime.strptime(msg_row.created_at, "%Y-%m-%d %H:%M")
    user = ds.user_by_id.get(user_id)

    flag_otp = bool(OTP_REQUEST_PATTERN.search(text)) and not bool(OTP_NEGATION_PATTERN.search(text))
    flag_urgency_words = bool(URGENCY_WORDS.search(text))
    flag_account_risk_language = bool(ACCOUNT_RISK_LANGUAGE.search(text))

    ev = {
        "message_id": msg_row.message_id,
        "user_id": user_id,
        "conversation_type": msg_row.conversation_type,
        "media_type": msg_row.media_type if isinstance(msg_row.media_type, str) else "",
        "forwarded_count": int(msg_row.forwarded_count or 0),
        "is_dnd_now": _parse_dnd(user.do_not_disturb_window, now) if user is not None else False,
        "user_reports_30d": int(user.messages_reported_30d) if user is not None else 0,
        "contradictions": [],
    }

    # ---- group context (facts, no scores) ----
    ev["group_type"] = ""
    ev["group_muted_by_user"] = False
    ev["sender_role"] = ""
    sender_id = getattr(msg_row, "sender_user_id", None)
    if isinstance(msg_row.group_id, str) and msg_row.group_id:
        group = ds.group_by_id.get(msg_row.group_id)
        if group is not None:
            ev["group_type"] = group.group_type
        member = ds.membership.get((msg_row.group_id, user_id))
        if member is not None:
            ev["group_muted_by_user"] = bool(member.group_muted_by_user)
        if isinstance(sender_id, str) and sender_id:
            sm = ds.membership.get((msg_row.group_id, sender_id))
            if sm is not None:
                ev["sender_role"] = sm.role
                ev["sender_messages_sent_30d"] = int(sm.messages_sent_30d)
                ev["sender_replies_sent_30d"] = int(sm.replies_sent_30d)

    # ---- business context (facts, no scores) ----
    ev["business_verified"] = False
    ev["business_domain_mismatch"] = False
    ev["business_report_rate_30d"] = None  # raw ratio, not a trust score
    ev["user_opted_out_of_promotions"] = False
    ev["user_has_recent_business_activity_180d"] = False
    if isinstance(msg_row.business_id, str) and msg_row.business_id:
        biz = ds.business_by_id.get(msg_row.business_id)
        if biz is not None:
            ev["business_verified"] = bool(biz.verified)
            ev["business_domain_mismatch"] = biz.official_domain != biz.domain_used_by_sender
            ev["business_report_rate_30d"] = (
                round(biz.user_reports_30d / biz.messages_sent_30d, 4)
                if biz.messages_sent_30d else None
            )
            if ev["business_verified"] and biz.messages_sent_30d and \
               biz.user_reports_30d / biz.messages_sent_30d > 0.02:
                ev["contradictions"].append(
                    "business is verified but has an elevated report rate "
                    f"({biz.user_reports_30d}/{biz.messages_sent_30d} messages reported)"
                )
        rel = ds.user_business.get((user_id, msg_row.business_id))
        if rel is not None:
            ev["user_opted_out_of_promotions"] = bool(
                isinstance(rel.promotions_opted_out_at, str) and rel.promotions_opted_out_at
            )
            ev["user_has_recent_business_activity_180d"] = bool(
                rel.activity_count_180d and rel.activity_count_180d > 0
            )
            ev["why_user_knows_business"] = rel.why_user_knows_account

    # ---- source history: raw counts + diversified evidence, no weighting ----
    hist = []
    if isinstance(sender_id, str) and sender_id:
        hist = ds.history_by_sender.get((user_id, sender_id), [])
    elif isinstance(msg_row.group_id, str) and msg_row.group_id:
        hist = ds.history_by_group.get((user_id, msg_row.group_id), [])
    elif isinstance(msg_row.business_id, str) and msg_row.business_id:
        hist = ds.history_by_business.get((user_id, msg_row.business_id), [])

    ev["has_history"] = len(hist) > 0
    ev["history_count"] = len(hist)
    ev["evidence_quality"] = _evidence_quality(len(hist))

    current_flags = {"otp": flag_otp, "urgency": flag_urgency_words}
    picked = _pick_diversified_evidence(ds, hist, text, current_flags)

    opened = replied = dismissed = reported = 0
    evidence_items = []
    for row in picked:
        e = ds.event_by_message.get(row.message_id)
        item = {
            "message_id": row.message_id,
            "created_at": row.created_at,
            "text_snippet": (row.message_text or "")[:140] if isinstance(row.message_text, str) else "",
            "opened": bool(e.message_opened) if e is not None else None,
            "replied": bool(e.message_replied) if e is not None else None,
            "dismissed": bool(e.notification_dismissed) if e is not None else None,
            "reported": bool(e.message_reported) if e is not None else None,
        }
        evidence_items.append(item)
        if e is not None:
            opened += int(e.message_opened)
            replied += int(e.message_replied)
            dismissed += int(e.notification_dismissed)
            reported += int(e.message_reported)

    # raw counts across ALL history from this source (not just the 3 picked),
    # so the LLM sees the full pattern, not a cherry-picked sample
    all_opened = all_replied = all_dismissed = all_reported = 0
    for row in hist:
        e = ds.event_by_message.get(row.message_id)
        if e is not None:
            all_opened += int(e.message_opened)
            all_replied += int(e.message_replied)
            all_dismissed += int(e.notification_dismissed)
            all_reported += int(e.message_reported)

    ev["source_history_raw_counts"] = {
        "opened": all_opened, "replied": all_replied,
        "dismissed": all_dismissed, "reported": all_reported,
        "total_messages": len(hist),
    }
    ev["evidence_message_ids"] = [item["message_id"] for item in evidence_items] or []
    ev["evidence_items"] = evidence_items

    if ev.get("sender_role") == "admin" and all_reported + all_dismissed >= 2 and all_opened == 0 and all_replied == 0:
        ev["contradictions"].append(
            "sender is a group admin, but the user's actual reaction history to this "
            "source is entirely dismiss/report with no opens or replies"
        )

    # ---- hard-override risk flags (checked in code BEFORE the LLM call) ----
    # Note: business_domain_mismatch itself already comes from a structured
    # field (business_accounts.csv official_domain vs domain_used_by_sender),
    # not text matching -- more reliable than trying to regex a URL out of
    # the message.
    ev["flag_otp_request"] = flag_otp
    ev["flag_urgency_language"] = flag_urgency_words
    ev["flag_account_risk_language"] = flag_account_risk_language
    ev["hard_override_scam"] = flag_otp or (
        ev["business_domain_mismatch"] and flag_account_risk_language
    )

    return ev


if __name__ == "__main__":
    import json
    from loader import Dataset
    ds = Dataset()
    for i, row in enumerate(ds.messages.itertuples()):
        if i >= 3:
            break
        print(json.dumps(build_evidence(ds, row), indent=2, default=str))
        print()
