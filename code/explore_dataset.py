# to get the overall estimates from the dataset

from collections import Counter
from loader import Dataset


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main():
    ds = Dataset()

    section("COUNTS")
    print(f"messages to route:  {len(ds.messages)}")
    print(f"users:               {len(ds.users)}")
    print(f"groups:              {len(ds.groups)}")
    print(f"businesses:          {len(ds.business_accounts)}")
    print(f"message_history rows:{len(ds.message_history)}")
    print(f"message_events rows: {len(ds.message_events)}")
    print(f"images:              {len(ds.images)}")
    print(f"voice_notes:         {len(ds.voice_notes)}")

    section("CONVERSATION TYPES (messages.csv)")
    print(Counter(ds.messages.conversation_type))

    section("MEDIA TYPES (messages.csv)")
    media = ds.messages.media_type.fillna("text")
    media = media.replace("", "text")
    print(Counter(media))

    section("MISSING VALUES (messages.csv)")
    print(ds.messages.isna().sum().to_string())

    section("COLD START: messages with ZERO usable history for their source")
    from evidence import build_evidence
    no_history = 0
    for row in ds.messages.itertuples():
        ev = build_evidence(ds, row)
        if not ev["has_history"]:
            no_history += 1
    print(f"{no_history} / {len(ds.messages)} messages have NO history for "
          f"their sender/group/business")

    section("HISTORY DEPTH DISTRIBUTION (per message, count of matching history rows)")
    depths = []
    for row in ds.messages.itertuples():
        ev = build_evidence(ds, row)
        depths.append(ev["history_count"])
    depths.sort()
    n = len(depths)
    print(f"min={depths[0]}  p25={depths[n//4]}  median={depths[n//2]}  "
          f"p75={depths[3*n//4]}  max={depths[-1]}")
    print(Counter(depths))

    section("BUSINESS_ID PRESENCE FOR business CONVERSATIONS")
    biz_convos = ds.messages[ds.messages.conversation_type == "business"]
    missing_biz_id = biz_convos.business_id.isna().sum()
    print(f"{missing_biz_id} / {len(biz_convos)} business-type messages missing business_id")

    section("GROUP_ID PRESENCE FOR group CONVERSATIONS")
    grp_convos = ds.messages[ds.messages.conversation_type == "group"]
    missing_grp_id = grp_convos.group_id.isna().sum()
    print(f"{missing_grp_id} / {len(grp_convos)} group-type messages missing group_id")

    section("FORWARDED COUNT DISTRIBUTION")
    print(ds.messages.forwarded_count.describe().to_string())

    section("BUSINESS VERIFICATION SPLIT")
    print(Counter(ds.business_accounts.verified))

    section("DOMAIN MISMATCH COUNT (official_domain != domain_used_by_sender)")
    mismatch = (ds.business_accounts.official_domain != ds.business_accounts.domain_used_by_sender).sum()
    print(f"{mismatch} / {len(ds.business_accounts)} businesses have a domain mismatch")

    section("USERS WITH NO GROUP MEMBERSHIPS OR BUSINESS HISTORY AT ALL")
    users_with_groups = set(ds.group_members.user_id)
    users_with_biz = set(ds.user_business_history.user_id)
    all_users = set(ds.users.user_id)
    isolated = all_users - users_with_groups - users_with_biz
    print(f"{len(isolated)} users with neither group membership nor business history: {isolated}")


if __name__ == "__main__":
    main()
