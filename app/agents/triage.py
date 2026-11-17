"""Triage agent — reads the user's request (with recent history) and decides:

- intent: "qa" (answer from documents), "summarize", or "smalltalk" (greeting /
  "what can you do", needs no documents)
- needs_retrieval: whether to hit the documents
- standalone_query: a follow-up like "what about clause 3?" rewritten into a
  self-contained query using the conversation so far (so retrieval works)

Uses the local LLM when available, with a deterministic rule-based fallback so the
agent still behaves sensibly (and tests run) without a model.
"""
from __future__ import annotations

import json
import re

INTENTS = ("qa", "summarize", "smalltalk")

_SYSTEM = (
    "You route a document assistant. Given the conversation and the latest user "
    "message, reply with ONLY a JSON object:\n"
    '{"intent": "qa|summarize|smalltalk", "needs_retrieval": true|false, '
    '"standalone_query": "<the user\'s question rewritten to stand alone>"}\n'
    "intent=smalltalk for greetings or questions about what you can do (no documents "
    "needed). intent=summarize when they ask for a summary/overview. Otherwise qa. "
    "Rewrite follow-ups into a standalone query using the history. No prose, JSON only."
)

_GREETING = re.compile(r"\b(hi|hello|hey|thanks|thank you|who are you|what can you do|"
                       r"what do you do|help)\b", re.I)
_SUMMARY = re.compile(r"(summar|overview|tl;?dr|gist|recap|key point)", re.I)


def _history_text(history: list[dict], limit: int = 6) -> str:
    turns = [h for h in history if h.get("role") in ("user", "assistant")][-limit:]
    return "\n".join(f"{h['role']}: {h['content']}" for h in turns)


def _rule_based(message: str, history: list[dict]) -> dict:
    m = message.strip()
    low = m.lower()
    if _GREETING.search(low) and len(low.split()) <= 7:
        return {"intent": "smalltalk", "needs_retrieval": False, "standalone_query": m}
    intent = "summarize" if _SUMMARY.search(low) else "qa"
    # cheap follow-up condensing: if the message is short and leans on context,
    # prepend the previous user question so retrieval has something concrete.
    standalone = m
    prior_users = [h["content"] for h in history if h.get("role") == "user"]
    if prior_users and (len(m.split()) <= 4 or low.startswith(("what about", "and ", "it ", "that "))):
        standalone = f"{prior_users[-1]} {m}"
    return {"intent": intent, "needs_retrieval": True, "standalone_query": standalone}


def _parse(raw: str) -> dict | None:
    lo, hi = raw.find("{"), raw.rfind("}")
    if lo == -1 or hi == -1:
        return None
    try:
        obj = json.loads(raw[lo:hi + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _normalize(obj: dict, message: str) -> dict:
    intent = obj.get("intent")
    if intent not in INTENTS:
        intent = "qa"
    needs = obj.get("needs_retrieval")
    needs = bool(needs) if isinstance(needs, bool) else (intent != "smalltalk")
    sq = obj.get("standalone_query")
    sq = sq if isinstance(sq, str) and sq.strip() else message
    return {"intent": intent, "needs_retrieval": needs, "standalone_query": sq.strip()}


def triage(message: str, history: list[dict] | None = None, llm=None) -> dict:
    """Return {intent, needs_retrieval, standalone_query}. Falls back to rules."""
    history = history or []
    if llm is None:
        return _rule_based(message, history)
    try:
        hist = _history_text(history)
        user = (f"Conversation so far:\n{hist}\n\n" if hist else "") + f"Latest user message: {message}"
        raw = llm.complete(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=200,
        )
        obj = _parse(raw)
        if obj is None:
            return _rule_based(message, history)
        return _normalize(obj, message)
    except Exception:
        # Any LLM/parse failure: still behave sensibly.
        return _rule_based(message, history)
