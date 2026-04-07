"""
api.py — FortiPrompt RedTeam Evaluator  •  FastAPI Backend
===========================================================
Exposes your existing evaluator pipeline as a REST API.

Endpoints
---------
POST /evaluate            — evaluate a plain (prompt, response) pair
POST /evaluate/openai     — evaluate a raw OpenAI ChatCompletion JSON
POST /evaluate/generate   — send a prompt to a target LLM, then evaluate the response
GET  /report              — compute + return metrics (ASR, TTB, RR, heatmap)
GET  /sessions            — list sessions (optional ?model= &attack= filters)
GET  /sessions/{id}       — get a single session with all its turns
GET  /health              — liveness probe

All endpoints return JSON.  Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from evaluator import MultiTurnEvaluator, SessionSummary
from judge_ensemble import JudgeVerdict
from llm_adapter import LLMAdapter, LLMAdapterConfig
from session_store import SessionStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

_evaluator: Optional[MultiTurnEvaluator] = None
_llm_adapter: Optional[LLMAdapter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _evaluator, _llm_adapter
    mongo_uri = os.getenv("HARMBENCH_MONGO_URI", "mongodb://localhost:27017")
    _evaluator = MultiTurnEvaluator(mongo_uri=mongo_uri, max_turns=int(os.getenv("MAX_TURNS", "10")))
    _llm_adapter = LLMAdapter(LLMAdapterConfig.from_env())
    logger.info("FortiPrompt API ready.")
    yield
    if _evaluator:
        _evaluator.close()
    logger.info("FortiPrompt API shut down.")


app = FastAPI(
    title="FortiPrompt RedTeam Evaluator",
    description="URT-Eval pipeline REST API for AI red-teaming and jailbreak evaluation.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins by default — restrict in production via CORS_ORIGINS env var
_allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    prompt:        str   = Field(..., description="The attacker/user prompt.")
    response:      str   = Field(..., description="The target LLM's response.")
    attack_method: str   = Field("", description="Attack method label (e.g. GCG, DirectRequest).")
    target_model:  str   = Field("", description="Target model identifier.")
    behavior_id:   str   = Field("", description="HarmBench behavior ID (optional).")
    session_id:    Optional[str] = Field(None, description="Reuse an existing session ID for multi-turn.")
    turn_index:    int   = Field(0, description="Zero-based turn index (multi-turn only).")
    is_benign:     bool  = Field(False, description="Mark this turn as a benign seed probe.")


class EvaluateOpenAIRequest(BaseModel):
    payload:       dict[str, Any] = Field(..., description="Raw OpenAI ChatCompletion JSON object.")
    attack_method: str            = Field("")
    target_model:  str            = Field("")
    behavior_id:   str            = Field("")
    session_id:    Optional[str]  = Field(None)
    turn_index:    int            = Field(0)
    is_benign:     bool           = Field(False)


class GenerateAndEvaluateRequest(BaseModel):
    """
    Send a prompt to the configured target LLM, capture the response,
    then run the full URT-Eval pipeline on the pair.
    """
    prompt:        str           = Field(..., description="Prompt to send to the target LLM.")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt override.")
    attack_method: str           = Field("")
    target_model:  str           = Field("", description="Override the adapter's default model.")
    behavior_id:   str           = Field("")
    session_id:    Optional[str] = Field(None)
    turn_index:    int           = Field(0)
    is_benign:     bool          = Field(False)
    # OpenAI ChatCompletion message history for multi-turn context
    messages:      Optional[list[dict[str, Any]]] = Field(
        None,
        description="Full prior message history (OpenAI format). "
                    "If provided, the prompt is appended as a new user turn."
    )


class EvaluateResponse(BaseModel):
    verdict:           str
    breach:            bool
    stage_reached:     int
    refusal_detected:  bool
    tool_calls_found:  bool
    latency_ms:        float
    session_id:        Optional[str]
    turn_index:        int
    labels: dict[str, Any] = Field(default_factory=dict)


class GenerateAndEvaluateResponse(EvaluateResponse):
    generated_response: str = ""


class SessionResponse(BaseModel):
    session_id:    str
    attack_method: str
    target_model:  str
    behavior_id:   str
    status:        str
    total_turns:   int
    breach_turn:   Optional[int]
    ttb:           Optional[int]
    breached:      bool
    turns:         list[dict[str, Any]] = Field(default_factory=list)


class ReportResponse(BaseModel):
    asr:     dict[str, Any]
    ttb:     dict[str, Any]
    rr:      dict[str, Any]
    heatmap: dict[str, Any]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _result_to_response(result, extra: dict | None = None) -> dict[str, Any]:
    d = result.to_dict()
    out = {
        "verdict":          d["verdict"],
        "breach":           result.verdict == JudgeVerdict.SUCCESS,
        "stage_reached":    d["stage_reached"],
        "refusal_detected": d["refusal_detected"],
        "tool_calls_found": d["tool_calls_found"],
        "latency_ms":       d["latency_ms"],
        "session_id":       d["session_id"],
        "turn_index":       d["turn_index"],
        "labels":           d["labels"],
    }
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "fortiprompt-evaluator"}


# ── POST /evaluate ────────────────────────────────────────────────────────

@app.post("/evaluate", response_model=EvaluateResponse, summary="Evaluate a prompt/response pair")
async def evaluate(req: EvaluateRequest):
    """
    Evaluate a single (prompt, response) pair through the URT-Eval pipeline.
    Stores the result in MongoDB.

    **Postman usage**: POST /evaluate with JSON body.
    """
    if _evaluator is None:
        raise HTTPException(503, "Evaluator not initialised")

    result = _evaluator._judge.evaluate(
        malicious_prompt=req.prompt,
        model_response=req.response,
        session_id=req.session_id,
        turn_index=req.turn_index,
    )

    # Persist to MongoDB
    if req.session_id:
        _evaluator._store.upsert_session(
            session_id=req.session_id,
            attack_method=req.attack_method,
            target_model=req.target_model,
            behavior_id=req.behavior_id,
        )
        _evaluator._store.record_turn(
            result, req.prompt, req.response, is_benign=req.is_benign
        )
        if result.verdict == JudgeVerdict.SUCCESS:
            _evaluator._store.mark_breached(req.session_id, req.turn_index)

    return _result_to_response(result)


# ── POST /evaluate/openai ─────────────────────────────────────────────────

@app.post("/evaluate/openai", response_model=EvaluateResponse,
          summary="Evaluate a raw OpenAI ChatCompletion object")
async def evaluate_openai(req: EvaluateOpenAIRequest):
    """
    Accepts a raw OpenAI `ChatCompletion` JSON (or `{messages:[...]}` shape).
    Step 0 log normalisation runs automatically.

    **Postman usage**: POST /evaluate/openai with the entire completion dict as the body.
    """
    if _evaluator is None:
        raise HTTPException(503, "Evaluator not initialised")

    result = _evaluator._judge.evaluate_openai(
        payload=req.payload,
        session_id=req.session_id,
        turn_index=req.turn_index,
    )

    if req.session_id:
        _evaluator._store.upsert_session(
            session_id=req.session_id,
            attack_method=req.attack_method,
            target_model=req.target_model,
            behavior_id=req.behavior_id,
        )
        exchange = _evaluator._extractor.extract(req.payload)
        _evaluator._store.record_turn(
            result, exchange.user_intent, exchange.target_output,
            is_benign=req.is_benign,
        )
        if result.verdict == JudgeVerdict.SUCCESS:
            _evaluator._store.mark_breached(req.session_id, req.turn_index)

    return _result_to_response(result)


# ── POST /evaluate/generate ───────────────────────────────────────────────

@app.post("/evaluate/generate", response_model=GenerateAndEvaluateResponse,
          summary="Send a prompt to a target LLM, then evaluate the response")
async def generate_and_evaluate(req: GenerateAndEvaluateRequest):
    """
    End-to-end single call:
    1. Sends the prompt to the configured target LLM (OpenAI, local, or SageMaker).
    2. Captures the response.
    3. Runs the full URT-Eval pipeline on the (prompt, response) pair.
    4. Persists to MongoDB and returns the evaluation result + the LLM response.

    Configure the LLM via environment variables — see `llm_adapter.py`.

    **Postman usage**: POST /evaluate/generate with JSON body.
    """
    if _evaluator is None or _llm_adapter is None:
        raise HTTPException(503, "Evaluator not initialised")

    try:
        generated_response = await _llm_adapter.generate(
            prompt=req.prompt,
            system_prompt=req.system_prompt,
            model_override=req.target_model or None,
            messages=req.messages,
        )
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        raise HTTPException(502, f"Target LLM error: {exc}") from exc

    result = _evaluator._judge.evaluate(
        malicious_prompt=req.prompt,
        model_response=generated_response,
        session_id=req.session_id,
        turn_index=req.turn_index,
    )

    if req.session_id:
        _evaluator._store.upsert_session(
            session_id=req.session_id,
            attack_method=req.attack_method,
            target_model=req.target_model or _llm_adapter.model_name,
            behavior_id=req.behavior_id,
        )
        _evaluator._store.record_turn(
            result, req.prompt, generated_response, is_benign=req.is_benign
        )
        if result.verdict == JudgeVerdict.SUCCESS:
            _evaluator._store.mark_breached(req.session_id, req.turn_index)

    return {**_result_to_response(result), "generated_response": generated_response}


# ── GET /report ───────────────────────────────────────────────────────────

@app.get("/report", response_model=ReportResponse, summary="Get evaluation metrics")
async def report(
    model:  Optional[str] = Query(None, description="Filter by target model name."),
    attack: Optional[str] = Query(None, description="Filter by attack method."),
):
    """
    Returns all four URT-Eval metrics as JSON:
    - **ASR** — Attack Success Rate
    - **TTB** — Turns-to-Breach distribution
    - **RR**  — Refusal Robustness
    - **heatmap** — Category × Attack-Method ASR matrix

    **Postman usage**: GET /report?model=gpt-4o&attack=GCG
    """
    if _evaluator is None:
        raise HTTPException(503, "Evaluator not initialised")

    from metrics_engine import MetricsEngine

    filters: dict[str, Any] = {}
    if model:
        filters["target_model"] = model
    if attack:
        filters["attack_method"] = attack

    turns           = _evaluator._store.all_turns(filters=filters if filters else None)
    breach_sessions = _evaluator._store.breach_turns()
    engine          = MetricsEngine(turns)

    asr     = engine.compute_asr()
    ttb     = engine.compute_ttb(breach_sessions)
    rr      = engine.compute_refusal_robustness()
    heatmap = engine.compute_heatmap()

    return {
        "asr": {
            "total_samples":          asr.total_samples,
            "successes":              asr.successes,
            "failures":               asr.failures,
            "hard_refusals":          asr.hard_refusals,
            "errors":                 asr.errors,
            "asr":                    asr.asr,
            "asr_pct":                asr.asr_pct,
            "intent_harmful_count":   asr.intent_harmful_count,
            "response_harmful_count": asr.response_harmful_count,
            "refusal_count":          asr.refusal_count,
        },
        "ttb": {
            "sessions_evaluated": ttb.sessions_evaluated,
            "sessions_breached":  ttb.sessions_breached,
            "mean_ttb":           ttb.mean_ttb,
            "median_ttb":         ttb.median_ttb,
            "min_ttb":            ttb.min_ttb,
            "max_ttb":            ttb.max_ttb,
            "ttb_distribution":   ttb.ttb_distribution,
        },
        "rr": {
            "benign_total":       rr.benign_total,
            "false_refusals":     rr.false_refusals,
            "refusal_robustness": rr.refusal_robustness,
        },
        "heatmap": {
            "categories":     heatmap.categories,
            "attack_methods": heatmap.attack_methods,
            "matrix":         heatmap.matrix,
        },
    }


# ── GET /sessions ─────────────────────────────────────────────────────────

@app.get("/sessions", summary="List all evaluation sessions")
async def list_sessions(
    model:  Optional[str] = Query(None),
    attack: Optional[str] = Query(None),
):
    """
    Returns all session documents.  Filter with `?model=` and/or `?attack=`.

    **Postman usage**: GET /sessions?attack=GCG
    """
    if _evaluator is None:
        raise HTTPException(503, "Evaluator not initialised")

    filters: dict[str, Any] = {}
    if model:
        filters["target_model"] = model
    if attack:
        filters["attack_method"] = attack

    sessions = _evaluator._store.get_all_sessions(filters=filters if filters else None)
    return {"sessions": sessions, "count": len(sessions)}


# ── GET /sessions/{session_id} ────────────────────────────────────────────

@app.get("/sessions/{session_id}", response_model=SessionResponse,
         summary="Get a session with all its turns")
async def get_session(session_id: str):
    """
    Returns the session document plus every turn recorded for it.

    **Postman usage**: GET /sessions/your-session-uuid
    """
    if _evaluator is None:
        raise HTTPException(503, "Evaluator not initialised")

    session = _evaluator._store.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id!r} not found")

    turns = _evaluator._store.all_turns(filters={"session_id": session_id})

    return {
        "session_id":    session.get("session_id", session_id),
        "attack_method": session.get("attack_method", ""),
        "target_model":  session.get("target_model", ""),
        "behavior_id":   session.get("behavior_id", ""),
        "status":        session.get("status", "UNKNOWN"),
        "total_turns":   session.get("total_turns", 0),
        "breach_turn":   session.get("breach_turn"),
        "ttb":           session.get("breach_turn"),
        "breached":      session.get("status") == "BREACHED",
        "turns":         turns,
    }
