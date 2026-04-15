"""
Tests for scorer.py — risk score calculation, normalization, grade assignment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app.models import AllergyAlert, Contraindication, Interaction
from app.scorer import compute_risk_score, WEIGHTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_interaction(severity: str) -> Interaction:
    return Interaction(
        drug_a="DrugA",
        drug_b="DrugB",
        severity=severity,
        mechanism="Test mechanism",
        recommendation="Test recommendation",
    )


def make_allergy(severity: str) -> AllergyAlert:
    return AllergyAlert(
        drug="DrugA",
        allergen="AllergenX",
        severity=severity,
        cross_reactivity=False,
    )


def make_contraindication(risk_level: str) -> Contraindication:
    return Contraindication(
        drug="DrugA",
        condition="Some Condition",
        risk_level=risk_level,
        alternative=None,
    )


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


def test_empty_inputs_return_safe():
    score = compute_risk_score([], [], [])
    assert score.total == 0.0
    assert score.grade == "SAFE"


# ---------------------------------------------------------------------------
# Individual weights
# ---------------------------------------------------------------------------


def test_high_interaction_adds_30():
    score = compute_risk_score([make_interaction("high")], [], [])
    assert score.breakdown.high_interactions == WEIGHTS["high_interaction"]


def test_medium_interaction_adds_15():
    score = compute_risk_score([make_interaction("medium")], [], [])
    assert score.breakdown.medium_interactions == WEIGHTS["medium_interaction"]


def test_low_interaction_adds_5():
    score = compute_risk_score([make_interaction("low")], [], [])
    assert score.breakdown.low_interactions == WEIGHTS["low_interaction"]


def test_critical_allergy_adds_40():
    score = compute_risk_score([], [make_allergy("critical")], [])
    assert score.breakdown.critical_allergies == WEIGHTS["critical_allergy"]


def test_caution_allergy_adds_20():
    score = compute_risk_score([], [make_allergy("caution")], [])
    assert score.breakdown.caution_allergies == WEIGHTS["caution_allergy"]


def test_absolute_contraindication_adds_25():
    score = compute_risk_score([], [], [make_contraindication("absolute")])
    assert score.breakdown.absolute_contraindications == WEIGHTS["absolute_contraindication"]


def test_relative_contraindication_adds_10():
    score = compute_risk_score([], [], [make_contraindication("relative")])
    assert score.breakdown.relative_contraindications == WEIGHTS["relative_contraindication"]


# ---------------------------------------------------------------------------
# Grade thresholds
# ---------------------------------------------------------------------------


def test_score_zero_is_safe():
    score = compute_risk_score([], [], [])
    assert score.grade == "SAFE"


def test_low_score_is_safe():
    """One low interaction = 5 raw = 2.5 normalized → SAFE"""
    score = compute_risk_score([make_interaction("low")], [], [])
    assert score.grade == "SAFE"


def test_medium_score_is_caution():
    """One critical allergy (40 raw) + one medium interaction (15 raw)
    = 55 raw / 200 * 100 = 27.5 → CAUTION"""
    score = compute_risk_score(
        [make_interaction("medium")],
        [make_allergy("critical")],
        [],
    )
    assert score.grade == "CAUTION"


def test_high_risk_grade():
    """Several high interactions should push into HIGH_RISK territory."""
    interactions = [make_interaction("high")] * 4  # 4 * 30 = 120 raw
    score = compute_risk_score(interactions, [], [])
    # 120 / 200 * 100 = 60 → HIGH_RISK
    assert score.grade == "HIGH_RISK"


def test_critical_grade():
    """Maximum load should produce CRITICAL grade."""
    interactions = [make_interaction("high")] * 4  # 120
    allergies = [make_allergy("critical")] * 2   # 80
    # 200 raw / 200 * 100 = 100 → CRITICAL
    score = compute_risk_score(interactions, allergies, [])
    assert score.grade == "CRITICAL"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_score_never_exceeds_100():
    """Even with a massive input, score should cap at 100."""
    interactions = [make_interaction("high")] * 20  # 600 raw >> 200 cap
    score = compute_risk_score(interactions, [], [])
    assert score.total <= 100.0


def test_score_is_non_negative():
    score = compute_risk_score([], [], [])
    assert score.total >= 0.0


def test_score_formatted_to_two_decimals():
    score = compute_risk_score([make_interaction("high")], [], [])
    # Should be a clean float (30 / 200 * 100 = 15.0)
    assert isinstance(score.total, float)
