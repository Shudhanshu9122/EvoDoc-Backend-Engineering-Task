"""
EvoDoc — Clinical Drug Safety Engine
Data Models (Pydantic v2)

Defines all request/response schemas with field-level validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class PatientContext(BaseModel):
    """Encapsulates patient-specific clinical information for a request."""

    age: int = Field(..., ge=0, le=130, description="Patient age in years")
    weight_kg: float = Field(..., gt=0, le=500, description="Patient weight in kilograms")
    conditions: list[str] = Field(
        default_factory=list,
        description="Active medical conditions (e.g. 'diabetes', 'hypertension')",
    )
    current_medications: list[str] = Field(
        default_factory=list,
        description="Drugs the patient is currently taking",
    )
    allergies: list[str] = Field(
        default_factory=list,
        description="Known drug allergens or drug classes",
    )

    @field_validator("conditions", "current_medications", "allergies", mode="before")
    @classmethod
    def normalize_list_strings(cls, v: list) -> list:
        """Strip whitespace and lower-case all list entries."""
        return [item.strip().lower() for item in v if item and item.strip()]


class EvaluationRequest(BaseModel):
    """Incoming evaluation request payload."""

    medicines: list[str] = Field(
        ...,
        min_length=1,
        description="Drugs to evaluate for interactions (at least 1 required)",
    )
    patient_history: PatientHistory

    @field_validator("medicines", mode="before")
    @classmethod
    def normalize_drugs(cls, v: list) -> list:
        """Normalize drug names: strip, deduplicate (case-insensitive), title-case."""
        seen: set[str] = set()
        result: list[str] = []
        for drug in v:
            if not drug or not drug.strip():
                continue
            normalized = drug.strip().title()
            key = normalized.lower()
            if key not in seen:
                seen.add(key)
                result.append(normalized)
        return result

    @model_validator(mode="after")
    def at_least_one_drug(self) -> "EvaluationRequest":
        if not self.medicines:
            raise ValueError("At least one valid medicine name is required.")
        return self


# ---------------------------------------------------------------------------
# Patient History Model
# ---------------------------------------------------------------------------


class PatientHistory(BaseModel):
    """
    Full clinical history of a patient.
    Provides richer context for personalized safety evaluation.
    """

    patient_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the patient record",
    )
    name: str | None = Field(
        default=None,
        description="Patient full name (optional)",
    )
    age: int = Field(..., ge=0, le=130, description="Age in years")
    weight_kg: float = Field(..., gt=0, le=500, description="Weight in kilograms")
    gender: Literal["male", "female", "other"] | None = Field(
        default=None,
        description="Patient gender",
    )

    # --- Clinical background ---
    conditions: list[str] = Field(
        default_factory=list,
        description="Active medical conditions e.g. 'diabetes', 'hypertension', 'ckd'",
    )
    allergies: list[str] = Field(
        default_factory=list,
        description="Known drug or substance allergies e.g. 'penicillin', 'sulfa'",
    )
    current_medications: list[str] = Field(
        default_factory=list,
        description="All drugs currently being taken by the patient",
    )
    past_medications: list[str] = Field(
        default_factory=list,
        description="Previously used medications (historical context)",
    )
    past_adverse_reactions: list[str] = Field(
        default_factory=list,
        description="Documented prior adverse drug reactions",
    )

    # --- Organ function (affects drug metabolism/dosing) ---
    renal_impairment: bool = Field(
        default=False,
        description="True if patient has known renal (kidney) impairment",
    )
    hepatic_impairment: bool = Field(
        default=False,
        description="True if patient has known hepatic (liver) impairment",
    )
    pregnancy: bool = Field(
        default=False,
        description="True if patient is currently pregnant",
    )

    # --- Record metadata ---
    recorded_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when this history record was created",
    )

    @field_validator(
        "conditions", "allergies", "current_medications",
        "past_medications", "past_adverse_reactions",
        mode="before",
    )
    @classmethod
    def normalize_list_strings(cls, v: list) -> list:
        return [item.strip().lower() for item in v if item and item.strip()]


# ---------------------------------------------------------------------------
# Response Sub-models
# ---------------------------------------------------------------------------


class Interaction(BaseModel):
    """
    Represents a pairwise drug-drug interaction finding.
    Matches the schema: interactions[{drug_a, drug_b, severity,
    mechanism, clinical_recommendation, source_confidence}]
    """

    drug_a: str = Field(description="First drug in the interaction pair")
    drug_b: str = Field(description="Second drug in the interaction pair")
    severity: Literal["high", "medium", "low"] = Field(
        description="Clinical severity of the interaction"
    )
    mechanism: str = Field(
        description="Pharmacological mechanism behind the interaction"
    )
    clinical_recommendation: str = Field(
        description="Actionable recommendation for the prescribing clinician"
    )
    source_confidence: str = Field(
        description=(
            "Confidence level: e.g. 'high (clinical rule)', "
            "'medium (LLM-inferred)', 'low (heuristic)'"
        )
    )


class AllergyAlert(BaseModel):
    """
    Represents an allergy or cross-reactivity warning.
    Matches the schema: allergy_alerts[{medicine, reason, severity}]
    """

    medicine: str = Field(
        description="The medication that triggered the allergy alert"
    )
    reason: str = Field(
        description=(
            "Why this medication is flagged — direct allergy match "
            "or drug-class cross-reactivity"
        )
    )
    severity: str = Field(
        description="Severity of the alert: 'critical', 'caution', or 'warning'"
    )


class Contraindication(BaseModel):
    """Describes a drug-condition contraindication for this patient."""

    drug: str = Field(description="Drug that is contraindicated")
    condition: str = Field(description="Patient condition causing the contraindication")
    risk_level: Literal["absolute", "relative"] = Field(
        description="Absolute = must not use; Relative = use with caution"
    )
    alternative: str | None = Field(
        default=None,
        description="Suggested safer alternative drug, if available",
    )


class RiskBreakdown(BaseModel):
    """Detailed breakdown of score contributions by finding type."""

    high_interactions: float = 0.0
    medium_interactions: float = 0.0
    low_interactions: float = 0.0
    critical_allergies: float = 0.0
    caution_allergies: float = 0.0
    absolute_contraindications: float = 0.0
    relative_contraindications: float = 0.0


class RiskScore(BaseModel):
    """Final normalized risk score with grade and detailed breakdown."""

    total: float = Field(ge=0.0, le=100.0, description="Normalized risk score 0–100")
    grade: Literal["low", "medium", "high"]
    breakdown: RiskBreakdown


# ---------------------------------------------------------------------------
# Main Response Model — matches specified API schema exactly
# ---------------------------------------------------------------------------


class EvaluationResponse(BaseModel):
    """
    Complete drug safety evaluation response.

    Schema:
        interactions       → list of drug-drug interaction findings
        allergy_alerts     → list of allergy/cross-reactivity warnings
        safe_to_prescribe  → bool (False if any HIGH interaction or CRITICAL allergy)
        overall_risk_level → str (SAFE / CAUTION / HIGH_RISK / CRITICAL)
        requires_doctor_review → bool
        source             → 'llm' | 'fallback' | 'cache'
        cache_hit          → bool
        processing_time_ms → int
    """

    interactions: list[Interaction] = Field(
        default_factory=list,
        description="All detected drug-drug interactions",
    )
    allergy_alerts: list[AllergyAlert] = Field(
        default_factory=list,
        description="Allergy and cross-reactivity warnings",
    )
    safe_to_prescribe: bool = Field(
        description=(
            "True if no HIGH severity interaction or CRITICAL allergy detected. "
            "False means prescribing carries significant risk."
        )
    )
    overall_risk_level: str = Field(
        description="Overall risk grade: low / medium / high"
    )
    requires_doctor_review: bool = Field(
        description=(
            "True if any HIGH severity interaction, CRITICAL allergy, "
            "or ABSOLUTE contraindication was found"
        )
    )
    source: str = Field(
        description="Data source: 'llm' (AI-generated), 'fallback' (rule engine), or 'cache'"
    )
    cache_hit: bool = Field(
        description="True if this response was served from cache (no computation done)"
    )
    processing_time_ms: int = Field(
        description="Total wall-clock time to generate the response in milliseconds"
    )


# ---------------------------------------------------------------------------
# Operational / Utility Models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response for the GET /api/v1/health endpoint."""

    status: Literal["ok", "degraded"]
    uptime_seconds: float
    cache_size: int
    version: str = "1.0.0"


class CacheStats(BaseModel):
    """Cache performance metrics for GET /api/v1/cache/stats."""

    total_keys: int
    hits: int
    misses: int
    hit_rate_percent: float
    oldest_entry_age_seconds: float | None
