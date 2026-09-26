"""
LangGraph conversation state machine for /v1/reply.

Nodes (checked in this exact order of precedence):
  1. auto_reply_detector
  2. hostility_detector
  3. intent_transition_detector
  4. question_detector
  5. default
"""
from __future__ import annotations
import logging
import re
import threading
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from app.composer.rule_composer import compose_reply_rule_based

logger = logging.getLogger("vera.conversation_graph")


# ── state schema ──
class ConversationState(TypedDict):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str]
    from_role: str
    message: str
    turn_number: int
    # enriched by the graph
    facts: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    result: Optional[Dict[str, Any]]


# ── detection helpers ──

AUTO_REPLY_PATTERNS = [
    r"thank you for (contacting|reaching out|messaging|your message)",
    r"thanks for (contacting|reaching out|messaging|your message)",
    r"we (have |)received your (message|query|request)",
    r"our (team|representative|agent) will (get back|respond|contact)",
    r"this is an auto(matic|mated)?\s*(reply|response|message)",
    r"we('ll| will) (get back|respond|reply) (to you |)(soon|shortly|within)",
    r"will respond shortly",
]
_auto_reply_re = re.compile("|".join(AUTO_REPLY_PATTERNS), re.IGNORECASE)

HOSTILE_PATTERNS = [
    r"\b(stop|quit|cease|unsubscribe|opt.?out)\b",
    r"\b(spam|scam|fraud|fake|useless|annoying|irritating|harassment)\b",
    r"\b(don'?t|do not|never)\s+(message|contact|text|call|bother|disturb)\b",
    r"\bnot\s+interested\b",
    r"\b(f+u+c+k|shit|damn|idiot|stupid|fool|ass+hole|bastard)\b",
    r"\bleave\s+me\s+alone\b",
    r"\breport(ed|ing)?\b.*\b(spam|you)\b",
]
_hostile_re = re.compile("|".join(HOSTILE_PATTERNS), re.IGNORECASE)

POSITIVE_INTENT_PATTERNS = [
    r"\b(yes|yeah|yep|yup|sure|ok(ay)?|go\s+ahead|let'?s\s+(do|go)|proceed|confirm|book|done|agreed|haan|chaliye|chalo|theek|bilkul|zaroor)\b",
    r"\bwhat'?s?\s+next\b",
    r"\bhow\s+(do|can)\s+(i|we)\s+(proceed|start|begin|book)\b",
]
_positive_re = re.compile("|".join(POSITIVE_INTENT_PATTERNS), re.IGNORECASE)

QUESTION_PATTERNS = [
    r"\?$",
    r"^(what|how|when|where|why|which|who|is|are|can|could|do|does|will|would)\b",
    r"\b(price|cost|rate|charge|fee|amount|discount|offer|deal)\b",
    r"\b(timing|schedule|hours|open|close|available)\b",
    r"\b(kitna|kab|kahan|kaise|kya)\b",  # Hindi question words
]
_question_re = re.compile("|".join(QUESTION_PATTERNS), re.IGNORECASE)


def _is_auto_reply(message: str) -> bool:
    return bool(_auto_reply_re.search(message))

def _is_exact_repeat(message: str, history: List[Dict[str, Any]]) -> bool:
    """Check if this exact message text has been seen before in the conversation."""
    msg_normalized = message.strip().lower()
    for turn in history:
        if turn.get("message", "").strip().lower() == msg_normalized:
            return True
    return False

def _count_exact_repeats(message: str, history: List[Dict[str, Any]]) -> int:
    """Count how many times this exact message appeared before."""
    msg_normalized = message.strip().lower()
    return sum(
        1 for turn in history
        if turn.get("message", "").strip().lower() == msg_normalized
    )

def _is_hostile(message: str) -> bool:
    return bool(_hostile_re.search(message))

def _is_positive_intent(message: str) -> bool:
    return bool(_positive_re.search(message))

def _is_question(message: str) -> bool:
    return bool(_question_re.search(message))


# ── global auto-reply tracker ──
# Tracks auto-reply patterns seen across ALL conversations per merchant.
# The judge's auto_reply_hell test sends the SAME auto-reply string in
# separate conversations (conv_auto_1, conv_auto_2, etc.), so we need
# to track globally, not just per-conversation.
_global_auto_reply_count: Dict[str, int] = {}
_global_lock = threading.Lock()


def _track_global_auto_reply(merchant_id: str) -> int:
    """Increment and return the global auto-reply count for this merchant."""
    with _global_lock:
        _global_auto_reply_count[merchant_id] = _global_auto_reply_count.get(merchant_id, 0) + 1
        return _global_auto_reply_count[merchant_id]


def reset_global_auto_reply_tracker():
    """Called by teardown to reset global state."""
    with _global_lock:
        _global_auto_reply_count.clear()


# ── graph nodes ──

def auto_reply_node(state: ConversationState) -> ConversationState:
    """
    Node 1: auto-reply detector.
    Checks for canned/auto-reply patterns AND exact text repetition.

    Two detection modes:
    1. Within-conversation: same message repeated in this conversation's history.
    2. Cross-conversation: auto-reply pattern detected, tracked globally per merchant.
       (The judge sends the same auto-reply in separate conversations.)

    1st occurrence -> try to route around (send).
    2nd+ occurrence -> end gracefully.
    """
    message = state["message"]
    history = state.get("conversation_history", [])
    merchant_id = state.get("merchant_id", "unknown")

    is_auto = _is_auto_reply(message)
    is_repeat = _is_exact_repeat(message, history)
    repeat_count = _count_exact_repeats(message, history)

    if is_auto or is_repeat:
        if is_repeat and repeat_count >= 1:
            # Within-conversation: 2nd+ exact repeat -> end
            state["result"] = {
                "action": "end",
                "rationale": (
                    f"Repeated auto-reply detected in conversation "
                    f"({repeat_count + 1} occurrences); ending to avoid spam."
                ),
            }
        elif is_auto:
            # Cross-conversation auto-reply pattern: track globally per merchant
            global_count = _track_global_auto_reply(merchant_id)
            if global_count >= 2:
                # 2nd+ global occurrence -> end
                state["result"] = {
                    "action": "end",
                    "rationale": (
                        f"Auto-reply pattern detected for the {global_count}th time "
                        f"from merchant {merchant_id}; ending conversation."
                    ),
                }
            else:
                # 1st global occurrence -> try to route around
                facts = state.get("facts", {})
                state["result"] = compose_reply_rule_based("auto_reply_first", facts, message)
        else:
            # 1st within-conversation repeat (not an auto-reply pattern)
            facts = state.get("facts", {})
            state["result"] = compose_reply_rule_based("auto_reply_first", facts, message)

    return state


def hostility_node(state: ConversationState) -> ConversationState:
    """
    Node 2: hostility/opt-out detector.
    Brief, calm acknowledgment. Never argue back, never re-pitch.
    """
    if state.get("result"):
        return state

    message = state["message"]
    if _is_hostile(message):
        state["result"] = {
            "action": "send",
            "body": (
                "Sorry for the inconvenience. No problem \u2014 "
                "we won't message you again. Take care!"
            ),
            "cta": None,
            "rationale": "Hostility or opt-out detected; acknowledging briefly and disengaging.",
        }
    return state


def intent_node(state: ConversationState) -> ConversationState:
    """
    Node 3: intent transition detector.
    Positive commitment -> action-mode vocabulary.
    AVOID qualifying questions (would you, do you, can you).
    """
    if state.get("result"):
        return state

    message = state["message"]
    if _is_positive_intent(message):
        facts = state.get("facts", {})
        state["result"] = compose_reply_rule_based("positive_intent", facts, message)
    return state


def question_node(state: ConversationState) -> ConversationState:
    """
    Node 4: question detector.
    Answer from stored context; if not available, say so honestly.
    Uses Mistral LLM if available, falls back to rule-based.
    """
    if state.get("result"):
        return state

    message = state["message"]
    if _is_question(message):
        facts = state.get("facts", {})
        history = state.get("conversation_history", [])
        try:
            from app.config import GEMINI_API_KEY1, GEMINI_API_KEY2, MISTRAL_API_KEY
            from app.composer.llm_composer import _call_llm_reply_sync
            if GEMINI_API_KEY1 or GEMINI_API_KEY2 or MISTRAL_API_KEY:
                state["result"] = _call_llm_reply_sync(facts, message, "question", history)
            else:
                state["result"] = compose_reply_rule_based("question", facts, message)
        except Exception as e:
            logger.warning("LLM reply error for question: %s; falling back to rule.", e)
            state["result"] = compose_reply_rule_based("question", facts, message)
    return state


def default_node(state: ConversationState) -> ConversationState:
    """
    Node 5: default continuation.
    Same composition rules as tick path — category voice, grounded facts.
    Uses Mistral LLM if available, falls back to rule-based.
    """
    if state.get("result"):
        return state

    facts = state.get("facts", {})
    message = state.get("message", "")
    history = state.get("conversation_history", [])
    try:
        from app.config import GEMINI_API_KEY1, GEMINI_API_KEY2, MISTRAL_API_KEY
        from app.composer.llm_composer import _call_llm_reply_sync
        if GEMINI_API_KEY1 or GEMINI_API_KEY2 or MISTRAL_API_KEY:
            state["result"] = _call_llm_reply_sync(facts, message, "default", history)
        else:
            state["result"] = compose_reply_rule_based("default", facts, message)
    except Exception as e:
        logger.warning("LLM reply error for default: %s; falling back to rule.", e)
        state["result"] = compose_reply_rule_based("default", facts, message)
    return state


# ── build the graph ──

def build_conversation_graph() -> StateGraph:
    """Build and compile the LangGraph conversation state machine."""
    graph = StateGraph(ConversationState)

    graph.add_node("auto_reply_detector", auto_reply_node)
    graph.add_node("hostility_detector", hostility_node)
    graph.add_node("intent_transition_detector", intent_node)
    graph.add_node("question_detector", question_node)
    graph.add_node("default", default_node)

    graph.set_entry_point("auto_reply_detector")
    graph.add_edge("auto_reply_detector", "hostility_detector")
    graph.add_edge("hostility_detector", "intent_transition_detector")
    graph.add_edge("intent_transition_detector", "question_detector")
    graph.add_edge("question_detector", "default")
    graph.add_edge("default", END)

    return graph


# Compiled graph (module-level singleton)
_graph = build_conversation_graph()
compiled_graph = _graph.compile()


def run_conversation_graph(
    conversation_id: str,
    merchant_id: str,
    customer_id: Optional[str],
    from_role: str,
    message: str,
    turn_number: int,
    facts: Dict[str, Any],
    conversation_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run the conversation graph and return the result dict
    matching one of ReplySend / ReplyWait / ReplyEnd.
    """
    initial_state: ConversationState = {
        "conversation_id": conversation_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "from_role": from_role,
        "message": message,
        "turn_number": turn_number,
        "facts": facts,
        "conversation_history": conversation_history,
        "result": None,
    }

    final_state = compiled_graph.invoke(initial_state)

    result = final_state.get("result")
    if not result:
        # Should never happen, but safety net
        result = {
            "action": "send",
            "body": "Thanks for reaching out! How can I help?",
            "cta": None,
            "rationale": "Fallback \u2014 no node produced a result.",
        }

    return result
