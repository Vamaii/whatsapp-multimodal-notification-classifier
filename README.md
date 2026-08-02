# WhatsApp Multimodal Notification Classifier

A multimodal notification routing pipeline built for the **HackerRank Orchestrate (August 2026)** hackathon.

The goal is to classify WhatsApp messages into one of three actions:

- 🔔 Notify
- 📰 Digest
- 🔕 Mute

The project handles text, images, and voice notes by converting every message into a common representation before making a final routing decision.

> **Note**
> This project was not submitted before the hackathon deadline due to free-tier API quota limits and last min network, charging issues. The implementation is published here as a record of the work and the design decisions that went into it.

---

# Features

- Text, image, and voice message support
- Unified multimodal processing pipeline
- Evidence-based personalization using user history
- Provider-independent LLM interface
- Prompt injection hardening
- Decision validation layer
- Detailed debugging and tracing support

---

# Project Structure

```
.
├── code/
│   ├── loader.py
│   ├── evidence.py
│   ├── media.py
│   ├── prompt_builder.py
│   ├── decision.py
│   ├── validator.py
│   ├── reasoners.py
│   ├── llm_clients.py
│   ├── debug_logger.py
│   └── main.py
│
├── dataset/
│
└── README.md
```

---

# How it Works

```
CSV Dataset
      │
      ▼
 loader.py
      │
      ▼
 evidence.py
      │
      ▼
 media.py
      │
      ▼
 prompt_builder.py
      │
      ▼
 decision.py
      │
      ▼
 validator.py
      │
      ▼
 output.csv
```

The pipeline separates deterministic processing from LLM reasoning.

Python is responsible for gathering evidence, validating inputs, and applying hard safety rules. The language model only performs the final reasoning step using the structured evidence provided to it.

---

# Design Decisions

## Decision Model

The router does not rely on manually written rules such as:

> "If the sender is X, always notify."

Instead, it builds an **evidence card** from historical interactions and lets the reasoning model make the final decision.

For every incoming message, Python computes objective evidence such as:

- Number of messages previously opened
- Replies sent
- Messages dismissed
- Messages reported
- Sender verification
- Domain mismatch
- User history with the sender
- Similar historical messages
- Evidence quality (amount of historical data)
- Contradictory signals

Earlier versions attempted to combine these into a single engagement score:

```
engagement =
0.4 × opened +
0.4 × replied -
0.8 × reported -
0.2 × dismissed
```

The more we tested it, the less we liked it.

Those weights were arbitrary—they weren't learned from data or backed by any justification. Instead of pretending this number represented "importance", the final design passes the raw evidence to the LLM.

For example, instead of saying:

```
Engagement = 0.73
```

the model receives information like:

```
Sender history
--------------
Opened: 18
Replies: 7
Dismissed: 2
Reported: 0

Evidence quality: High

Contradictions:
None
```

The reasoning model is then responsible for interpreting these facts in context rather than relying on a handcrafted scoring formula.

This separation keeps deterministic processing in Python while leaving subjective judgment to the language model.


## A Small Design Change That Had a Big Impact

One change completely altered how the router makes decisions.

Instead of asking:

> "Who sent this message?"

the prompt asks:

> "What is the cost of ignoring this message?"

This shifts the focus from sender identity to consequence.

For example:

- A package delivery update may be important even if it's from an unfamiliar business.
- A frequently muted group message may still deserve a notification if it announces an emergency.
- An OTP phishing message should never be promoted simply because it appears urgent.

Thinking in terms of consequences produced much more consistent decisions than relying only on sender history.

# Running the Project

Clone the repository:

```bash
git clone https://github.com/Vamaii/whatsapp-multimodal-notification-classifier.git
cd whatsapp-multimodal-notification-classifier
```

Install dependencies:

```bash
pip install -r code/requirements.txt
```

Create a `.env` file containing the required API keys.

Run:

```bash
python code/main.py
```

Predictions are written to:

```
dataset/output.csv
```

---

# Tech Stack

- Python
- Pandas
- Gemini API
- Groq API
- OpenRouter
- OCR / Speech-to-Text (multimodal models)

---

# Current Status

The deterministic parts of the pipeline were tested against the provided dataset.

The LLM-based pipeline is implemented, but a complete end-to-end run could not be finished before the hackathon deadline because the available free-tier API quota was exhausted.

---

# License

This project is published for learning and portfolio purposes.
