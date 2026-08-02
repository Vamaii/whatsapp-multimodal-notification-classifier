"""
Loads all dataset/*.csv files and builds lookup indices so downstream
steps (features, retrieval) can join in O(1) instead of scanning
the full CSVs per message. yeahh so
"""
import os
import pandas as pd
from collections import defaultdict

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")


def _read(name):
    return pd.read_csv(os.path.join(DATASET_DIR, name))


class Dataset:
    def __init__(self):
        self.messages = _read("messages.csv")
        self.users = _read("users.csv")
        self.groups = _read("groups.csv")
        self.group_members = _read("group_members.csv")
        self.business_accounts = _read("business_accounts.csv")
        self.user_business_history = _read("user_business_history.csv")
        self.message_history = _read("message_history.csv")
        self.message_events = _read("message_events.csv")
        self.images = _read("images.csv")
        self.voice_notes = _read("voice_notes.csv")
        self.daily_notification_summary = _read("daily_notification_summary.csv")

        self._build_indices()

    def _build_indices(self):
        # user_id -> row
        self.user_by_id = {r.user_id: r for r in self.users.itertuples()}

        # group_id -> row
        self.group_by_id = {r.group_id: r for r in self.groups.itertuples()}

        # business_id -> row
        self.business_by_id = {
            r.business_id: r for r in self.business_accounts.itertuples()
        }

        # (group_id, user_id) -> membership row
        self.membership = {
            (r.group_id, r.user_id): r for r in self.group_members.itertuples()
        }

        # (user_id, business_id) -> relationship row
        self.user_business = {
            (r.user_id, r.business_id): r
            for r in self.user_business_history.itertuples()
        }

        # image_id / voice_note_id -> file_path
        self.image_path = dict(zip(self.images.image_id, self.images.file_path))
        self.voice_path = dict(
            zip(self.voice_notes.voice_note_id, self.voice_notes.file_path)
        )

        # message_id -> event row (message_events keyed by user_id+message_id, but
        # message_id is unique to a single historical message in this dataset)
        self.event_by_message = {
            r.message_id: r for r in self.message_events.itertuples()
        }

        # sender_user_id -> list of historical message rows (most recent first)
        by_sender = defaultdict(list)
        by_group = defaultdict(list)
        by_business = defaultdict(list)
        for r in self.message_history.itertuples():
            if isinstance(r.sender_user_id, str) and r.sender_user_id:
                by_sender[(r.user_id, r.sender_user_id)].append(r)
            if isinstance(r.group_id, str) and r.group_id:
                by_group[(r.user_id, r.group_id)].append(r)
            if isinstance(r.business_id, str) and r.business_id:
                by_business[(r.user_id, r.business_id)].append(r)

        def _sort(d):
            for k in d:
                d[k].sort(key=lambda row: row.created_at, reverse=True)
            return d

        self.history_by_sender = _sort(by_sender)
        self.history_by_group = _sort(by_group)
        self.history_by_business = _sort(by_business)

        # daily notification load: (user_id, date) -> row
        self.daily_load = {
            (r.user_id, r.date): r
            for r in self.daily_notification_summary.itertuples()
        }


if __name__ == "__main__":
    ds = Dataset()
    print(f"messages to route:      {len(ds.messages)}")
    print(f"users:                  {len(ds.users)}")
    print(f"groups:                 {len(ds.groups)}")
    print(f"businesses:             {len(ds.business_accounts)}")
    print(f"history rows:           {len(ds.message_history)}")
    print(f"event rows:             {len(ds.message_events)}")
    print(f"images / voice notes:   {len(ds.images)} / {len(ds.voice_notes)}")
