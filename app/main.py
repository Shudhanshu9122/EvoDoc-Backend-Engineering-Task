"""
EvoDoc — Clinical Drug Safety Engine
FastAPI Application Entry Point

Routes:
  POST /api/v1/evaluate       — Main evaluation endpoint
  GET  /api/v1/health         — Liveness check
  GET  /api/v1/cache/stats    — Cache performance
  DELETE /api/v1/cache/flush  — Admin: clear cache
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.cache import DrugSafetyCache, get_cache_client

from app.models import (
    CacheStats,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    RiskScore,
)
from app.scorer import compute_risk_score
from app.validator import normalize_drug_list

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("evodoc.api")

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_start_time: float = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time
    _start_time = time.monotonic()
    logger.info("EvoDoc Safety Engine initialized.")
    yield
    logger.info("EvoDoc Safety Engine shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EvoDoc Clinical Drug Safety Engine",
    description=(
        "Production-grade API for evaluating drug-drug interactions, "
        "allergy risks, contraindications, and risk scoring."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


from app.engine import analyze_drugs

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post(
    "/analyze",
    response_model=EvaluationResponse,
    summary="Evaluate drug combinations for a patient",
    tags=["Safety Evaluation"],
)
async def evaluate_drugs(
    request: EvaluationRequest,
) -> EvaluationResponse:
    """
    Evaluate the given drugs for:
    - Drug-drug interactions (pairwise)
    - Allergy and cross-reactivity risks
    - Patient-specific contraindications
    - Overall normalized risk score (0–100)

    Results are cached for 1 hour by drug combination.
    """
    return await analyze_drugs(request.medicines, request.patient_history)



@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["Operations"],
)
async def health_check(cache: DrugSafetyCache = Depends(get_cache_client)) -> HealthResponse:
    """Returns service liveness, uptime, and current cache size."""
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.monotonic() - _start_time, 2),
        cache_size=cache.size,
    )


@app.get(
    "/api/v1/cache/stats",
    response_model=CacheStats,
    summary="Cache performance metrics",
    tags=["Operations"],
)
async def cache_stats(cache: DrugSafetyCache = Depends(get_cache_client)) -> CacheStats:
    """Returns hit/miss counts, hit rate, and oldest entry age."""
    return cache.stats()


@app.delete(
    "/api/v1/cache/flush",
    summary="Flush all cache entries (admin)",
    tags=["Operations"],
)
async def flush_cache(cache: DrugSafetyCache = Depends(get_cache_client)) -> dict:
    """Clear all cached evaluations. Use for admin/testing purposes."""
    count = cache.flush()
    logger.warning("Cache flushed — %d entries removed.", count)
    return {"message": f"Cache flushed. {count} entries removed."}


# ---------------------------------------------------------------------------
# Dev server entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
