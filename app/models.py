from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
import uuid


# ── POST /v1/context ─────────────────────────────────────────────────────
class ContextRequest(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str

class ContextAccepted(BaseModel):
    accepted: bool = True
    ack_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stored_at: str

class ContextRejected(BaseModel):
    accepted: bool = False
    reason: str
    current_version: Optional[int] = None
    details: Optional[str] = None


# ── POST /v1/tick ────────────────────────────────────────────────────────
class TickRequest(BaseModel):
    now: str
    available_triggers: List[str]

class TickAction(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    send_as: str                             # "vera" | "merchant_on_behalf"
    trigger_id: str
    template_name: str
    template_params: List[str]
    body: str
    cta: Optional[str] = None
    suppression_key: str
    rationale: str

class TickResponse(BaseModel):
    actions: List[TickAction]


# ── POST /v1/reply ───────────────────────────────────────────────────────
class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: int

class ReplySend(BaseModel):
    action: Literal["send"] = "send"
    body: str
    cta: Optional[str] = None
    rationale: str

class ReplyWait(BaseModel):
    action: Literal["wait"] = "wait"
    wait_seconds: int
    rationale: str

class ReplyEnd(BaseModel):
    action: Literal["end"] = "end"
    rationale: str

ReplyResponse = Union[ReplySend, ReplyWait, ReplyEnd]


# ── GET /v1/healthz ──────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"
    uptime_seconds: int
    contexts_loaded: Dict[str, int]


# ── GET /v1/metadata ─────────────────────────────────────────────────────
class MetadataResponse(BaseModel):
    team_name: str
    team_members: List[str]
    model: str
    approach: str
    contact_email: str
    version: str
    submitted_at: str


# ── POST /v1/teardown ────────────────────────────────────────────────────
class TeardownResponse(BaseModel):
    wiped: bool = True
