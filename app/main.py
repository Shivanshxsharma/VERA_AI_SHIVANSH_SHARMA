"""
FastAPI app — Vera magicpin AI Challenge bot.
Registers all 6 endpoints (5 required + teardown).
"""
from __future__ import annotations
import logging
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import (
    BOOT_TIME, TEAM_NAME, TEAM_MEMBERS, MODEL_NAME,
    APPROACH, CONTACT_EMAIL, VERSION, SUBMITTED_AT,
    PAYLOAD_MAX_BYTES,
)
from app.models import (
    ContextRequest, ContextAccepted, ContextRejected,
    TickRequest, TickResponse,
    ReplyRequest,
    HealthResponse, MetadataResponse, TeardownResponse,
)
from app.store import store, VALID_SCOPES
from app.scheduler import process_tick
from app.composer.fact_sheet import build_fact_sheet
from app.graph.conversation_graph import run_conversation_graph, reset_global_auto_reply_tracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("vera.main")

app = FastAPI(title="Vera Bot", version=VERSION)


# ── POST /v1/context ─────────────────────────────────────────────────────
@app.post("/v1/context")
async def ingest_context(request: Request):
    # Payload size check (raw body)
    body_bytes = await request.body()
    if len(body_bytes) > PAYLOAD_MAX_BYTES:
        return JSONResponse(
            status_code=413,
            content={"accepted": False, "reason": "payload_too_large",
                     "details": f"Payload exceeds {PAYLOAD_MAX_BYTES} bytes."},
        )

    try:
        import json
        data = json.loads(body_bytes)
        ctx = ContextRequest(**data)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope",
                     "details": str(e)},
        )

    # Validate scope
    if ctx.scope not in VALID_SCOPES:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope",
                     "details": f"scope must be one of {sorted(VALID_SCOPES)}"},
        )

    # Version check — idempotent upsert
    current_version = store.get_version(ctx.scope, ctx.context_id)
    if current_version is not None and ctx.version <= current_version:
        return JSONResponse(
            status_code=409,
            content={
                "accepted": False,
                "reason": "stale_version",
                "current_version": current_version,
            },
        )

    # Store
    store.upsert(ctx.scope, ctx.context_id, ctx.version, ctx.payload, ctx.delivered_at)
    now_iso = datetime.now(timezone.utc).isoformat()
    logger.info(f"Context stored: scope={ctx.scope} id={ctx.context_id} v={ctx.version}")

    return JSONResponse(
        status_code=200,
        content={
            "accepted": True,
            "ack_id": str(uuid.uuid4()),
            "stored_at": now_iso,
        },
    )


# ── POST /v1/tick ────────────────────────────────────────────────────────
@app.post("/v1/tick")
async def tick(req: TickRequest):
    logger.info(f"Tick received: now={req.now} triggers={len(req.available_triggers)}")
    try:
        actions = process_tick(req.now, req.available_triggers)
    except Exception as e:
        logger.exception(f"Tick processing error: {e}")
        actions = []

    return TickResponse(actions=actions)


# ── POST /v1/reply ───────────────────────────────────────────────────────
@app.post("/v1/reply")
async def reply(req: ReplyRequest):
    logger.info(
        f"Reply received: conv={req.conversation_id} from={req.from_role} "
        f"turn={req.turn_number} msg={req.message[:80]!r}"
    )

    # Build facts from stored context
    merchant_entry = store.get("merchant", req.merchant_id)
    merchant_payload = merchant_entry["payload"] if merchant_entry else {}
    category_slug = ""
    if merchant_payload:
        identity = merchant_payload.get("identity", {})
        if isinstance(identity, dict):
            category_slug = identity.get("category_slug", "")
        if not category_slug:
            category_slug = merchant_payload.get("category_slug", "")
    category_entry = store.get("category", category_slug) if category_slug else None
    category_payload = category_entry["payload"] if category_entry else None

    customer_entry = store.get("customer", req.customer_id) if req.customer_id else None
    customer_payload = customer_entry["payload"] if customer_entry else None

    facts = build_fact_sheet(
        trigger_payload={},   # no trigger context in reply
        merchant_payload=merchant_payload,
        category_payload=category_payload,
        customer_payload=customer_payload,
    )

    # Get conversation history
    conversation_history = store.get_conversation(req.conversation_id)

    # Run conversation graph
    result = run_conversation_graph(
        conversation_id=req.conversation_id,
        merchant_id=req.merchant_id,
        customer_id=req.customer_id,
        from_role=req.from_role,
        message=req.message,
        turn_number=req.turn_number,
        facts=facts,
        conversation_history=conversation_history,
    )

    # Record this turn in history
    store.append_conversation_turn(req.conversation_id, {
        "from_role": req.from_role,
        "message": req.message,
        "turn_number": req.turn_number,
    })
    # Also record our response in history if we're sending
    if result.get("action") == "send" and result.get("body"):
        store.append_conversation_turn(req.conversation_id, {
            "from_role": "vera",
            "message": result["body"],
            "turn_number": req.turn_number,
        })

    logger.info(f"Reply result: action={result.get('action')} conv={req.conversation_id}")
    return JSONResponse(content=result)


# ── GET /v1/healthz ──────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status": "ok",
        "app": "VERA AI - magicpin AI Challenge",
        "team": "Vera_Shivansh",
        "endpoints": {
            "health": "/v1/healthz",
            "metadata": "/v1/metadata",
            "docs": "/docs",
            "context": "/v1/context",
            "tick": "/v1/tick",
            "reply": "/v1/reply",
            "teardown": "/v1/teardown"
        }
    }


@app.get("/healthz", response_model=HealthResponse)
@app.get("/v1/healthz", response_model=HealthResponse)
async def healthz():
    now = datetime.now(timezone.utc)
    uptime = int((now - BOOT_TIME).total_seconds())
    return HealthResponse(
        uptime_seconds=uptime,
        contexts_loaded={
            "category": store.count_by_scope("category"),
            "merchant": store.count_by_scope("merchant"),
            "customer": store.count_by_scope("customer"),
            "trigger": store.count_by_scope("trigger"),
        },
    )


# ── GET /v1/metadata ─────────────────────────────────────────────────────
@app.get("/metadata", response_model=MetadataResponse)
@app.get("/v1/metadata", response_model=MetadataResponse)
async def metadata():
    return MetadataResponse(
        team_name=TEAM_NAME,
        team_members=TEAM_MEMBERS,
        model=MODEL_NAME,
        approach=APPROACH,
        contact_email=CONTACT_EMAIL,
        version=VERSION,
        submitted_at=SUBMITTED_AT,
    )


# ── POST /v1/teardown ───────────────────────────────────────────────────
@app.post("/v1/teardown")
async def teardown():
    store.wipe()
    reset_global_auto_reply_tracker()
    try:
        from app.composer.llm_composer import rotator
        with rotator._lock:
            rotator.cooldowns.clear()
            rotator.current_gemini_idx = 0
    except Exception:
        pass
    logger.info("Teardown: all in-memory state wiped.")
    return TeardownResponse(wiped=True)


@app.post("/v1/debug/simulate_cooldown")
async def simulate_cooldown(key_idx: int = 0, seconds: float = 60.0):
    import time
    from app.composer.llm_composer import rotator
    with rotator._lock:
        rotator.cooldowns[key_idx] = time.time() + seconds
        rotator.current_gemini_idx = (key_idx + 1) % len(rotator.gemini_keys)
    return {"status": "cooldown_set", "key_idx": key_idx, "cooldown_until": rotator.cooldowns[key_idx]}

