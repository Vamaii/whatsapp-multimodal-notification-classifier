"""
Provider-agnostic reasoning interface. decision.py depends on this
interface, not on Grok or Gemini directly -- so swapping providers at the end its a matter of changing env and not the whole code
"""
from abc import ABC, abstractmethod
import json
from llm_clients import groq_reason, gemini_text, groq_reason


class ReasoningModel(ABC):
    @abstractmethod
    def decide_raw(self, system_prompt: str, user_prompt: str) -> str:
        """Return the raw text response (expected JSON string)."""
        ...


class GroqReasoner(ReasoningModel):
    def decide_raw(self, system_prompt: str, user_prompt: str) -> str:
        return groq_reason(system_prompt, user_prompt)


class GeminiReasoner(ReasoningModel):
    """Alternate reasoner for A/B comparison against GroqReasoner."""
    def decide_raw(self, system_prompt: str, user_prompt: str) -> str:
        return gemini_text(system_prompt, user_prompt)


def get_reasoner(name: str = None) -> ReasoningModel:
    """name: 'groq' (default) or 'gemini'. Reads REASONER env var if not passed."""
    import os
    choice = (name or os.environ.get("REASONER", "groq")).lower()
    if choice == "gemini":
        return GeminiReasoner()
    return GroqReasoner()
