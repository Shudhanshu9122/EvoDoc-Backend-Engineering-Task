"""
EvoDoc — Clinical Drug Safety Engine
Risk Scorer

Computes a normalized 0–100 risk score from interactions,
allergy alerts, and contraindications.
"""

from __future__ import annotations

from app.models import (
    AllergyAlert,
    Contraindication,
    Interaction,
    RiskBreakdown,
    RiskScore,
)

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

WEIGHTS = {
    "high_interaction": 30.0,
    "medium_interaction": 15.0,
    "low_interaction": 5.0,
    "critical_allergy": 40.0,
    "caution_allergy": 20.0,
    "absolute_contraindication": 25.0,
    "relative_contraindication": 10.0,
}

# Maximum theoretically possible score before normalization cap.
# Prevents aggressive clamping on small lists.
# We'll cap at 200 raw points → maps to 100 normalized.
MAX_RAW_SCORE = 200.0

# Grade thresholds (on normalized 0–100 scale)
GRADE_THRESHOLDS = [
    (60.0, "high"),
    (30.0, "medium"),
    (0.0, "low"),
]


# ---------------------------------------------------------------------------
# Public scorer
# ---------------------------------------------------------------------------


def compute_risk_score(
    interactions: list[Interaction],
    allergy_alerts: list[AllergyAlert],
    contraindications: list[Contraindication],
) -> RiskScore:
    """
    Accumulate weighted scores from all findings, normalize to 0–100,
    assign a clinical grade, and return a RiskScore model.
    """
    breakdown = RiskBreakdown()

    # --- Interactions ---
    for interaction in interactions:
        if interaction.severity == "high":
            breakdown.high_interactions += WEIGHTS["high_interaction"]
        elif interaction.severity == "medium":
            breakdown.medium_interactions += WEIGHTS["medium_interaction"]
        else:
            breakdown.low_interactions += WEIGHTS["low_interaction"]

    # --- Allergy alerts ---
    for alert in allergy_alerts:
        if alert.severity == "critical":
            breakdown.critical_allergies += WEIGHTS["critical_allergy"]
        else:
            breakdown.caution_allergies += WEIGHTS["caution_allergy"]

    # --- Contraindications ---
    for contraindication in contraindications:
        if contraindication.risk_level == "absolute":
            breakdown.absolute_contraindications += WEIGHTS["absolute_contraindication"]
        else:
            breakdown.relative_contraindications += WEIGHTS["relative_contraindication"]

    # --- Aggregate raw score ---
    raw_score = (
        breakdown.high_interactions
        + breakdown.medium_interactions
        + breakdown.low_interactions
        + breakdown.critical_allergies
        + breakdown.caution_allergies
        + breakdown.absolute_contraindications
        + breakdown.relative_contraindications
    )

    # --- Normalize to 0–100 ---
    normalized = min(raw_score / MAX_RAW_SCORE * 100.0, 100.0)
    normalized = round(normalized, 2)

    # --- Grade assignment ---
    grade = "SAFE"
    for threshold, label in GRADE_THRESHOLDS:
        if normalized >= threshold:
            grade = label
            break

    return RiskScore(total=normalized, grade=grade, breakdown=breakdown)
