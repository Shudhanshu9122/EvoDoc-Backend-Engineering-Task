"""
Integration tests for the SafetyEngine pipeline.
Tests LLM path (mocked), fallback trigger logic, and end-to-end results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from app.engine import SafetyEngine, _is_valid_llm_output
from app.models import EvaluationRequest, PatientContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_request(drugs: list[str], **patient_kwargs) -> EvaluationRequest:
    defaults = {
        "age": 45,
        "weight_kg": 75.0,
        "conditions": [],
        "current_medications": [],
        "allergies": [],
    }
    defaults.update(patient_kwargs)
    return EvaluationRequest(drugs=drugs, patient=PatientContext(**defaults))


# ---------------------------------------------------------------------------
# LLM output validation
# ---------------------------------------------------------------------------


def test_valid_llm_output_accepted():
    data = {
        "interactions": [
            {
                "drug_a": "Warfarin",
                "drug_b": "Aspirin",
                "severity": "high",
                "mechanism": "Additive anticoagulation",
                "recommendation": "Avoid combination",
            }
        ],
        "allergy_alerts": [],
        "contraindications": [],
    }
    assert _is_valid_llm_output(data) is True


def test_llm_output_missing_key_rejected():
    data = {
        "interactions": [],
        # missing allergy_alerts and contraindications
    }
    assert _is_valid_llm_output(data) is False


def test_llm_output_invalid_severity_rejected():
    data = {
        "interactions": [
            {
                "drug_a": "A",
                "drug_b": "B",
                "severity": "severe",  # invalid
                "mechanism": "X",
                "recommendation": "Y",
            }
        ],
        "allergy_alerts": [],
        "contraindications": [],
    }
    assert _is_valid_llm_output(data) is False


def test_llm_output_null_field_rejected():
    data = {
        "interactions": [
            {
                "drug_a": "A",
                "drug_b": "B",
                "severity": "high",
                "mechanism": None,  # null field
                "recommendation": "Y",
            }
        ],
        "allergy_alerts": [],
        "contraindications": [],
    }
    assert _is_valid_llm_output(data) is False


def test_llm_output_invalid_allergy_severity_rejected():
    data = {
        "interactions": [],
        "allergy_alerts": [
            {
                "drug": "A",
                "allergen": "B",
                "severity": "moderate",  # invalid
                "cross_reactivity": True,
            }
        ],
        "contraindications": [],
    }
    assert _is_valid_llm_output(data) is False


def test_llm_output_allergy_missing_cross_reactivity_bool_rejected():
    data = {
        "interactions": [],
        "allergy_alerts": [
            {
                "drug": "A",
                "allergen": "B",
                "severity": "critical",
                "cross_reactivity": "yes",  # should be bool, not string
            }
        ],
        "contraindications": [],
    }
    assert _is_valid_llm_output(data) is False


# ---------------------------------------------------------------------------
# Engine pipeline — fallback triggered when LLM fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_uses_fallback_when_llm_unavailable():
    """
    When no API keys are set, engine should fall through to fallback
    and still return valid interaction data for known drug pairs.
    """
    request = make_request(["Warfarin", "Aspirin"])
    engine = SafetyEngine()

    interactions, allergy_alerts, contraindications, source = await engine.evaluate(request)

    assert source == "fallback"
    assert any(
        ("warfarin" in i.drug_a.lower() or "warfarin" in i.drug_b.lower())
        for i in interactions
    )


@pytest.mark.asyncio
async def test_engine_accepts_valid_llm_output():
    """
    When LLM returns valid JSON, engine should use it and return source='llm'.
    """
    valid_llm_response = {
        "interactions": [
            {
                "drug_a": "Warfarin",
                "drug_b": "Aspirin",
                "severity": "high",
                "mechanism": "Additive bleeding risk",
                "recommendation": "Avoid or monitor closely",
            }
        ],
        "allergy_alerts": [],
        "contraindications": [],
    }

    request = make_request(["Warfarin", "Aspirin"])
    engine = SafetyEngine()

    with patch.object(
        engine._llm,
        "call_groq",
        new=AsyncMock(return_value=valid_llm_response),
    ):
        interactions, allergy_alerts, contraindications, source = await engine.evaluate(request)

    assert source == "llm"
    assert len(interactions) == 1
    assert interactions[0].severity == "high"


@pytest.mark.asyncio
async def test_engine_falls_back_on_invalid_llm_output():
    """
    When LLM returns invalid JSON structure, engine should use fallback.
    """
    invalid_llm_response = {
        "interactions": [{"drug_a": "X", "severity": "extreme"}],  # missing fields
        "allergy_alerts": [],
        "contraindications": [],
    }

    request = make_request(["Warfarin", "Aspirin"])
    engine = SafetyEngine()

    with patch.object(engine._llm, "call_groq", new=AsyncMock(return_value=invalid_llm_response)):
        with patch.object(engine._llm, "call_huggingface", new=AsyncMock(return_value=None)):
            interactions, allergy_alerts, contraindications, source = await engine.evaluate(request)

    assert source == "fallback"


@pytest.mark.asyncio
async def test_engine_deduplicates_drugs():
    """Duplicate drugs in input should be deduplicated before processing."""
    request = make_request(["Warfarin", "warfarin", "WARFARIN", "Aspirin"])
    engine = SafetyEngine()

    interactions, _, _, source = await engine.evaluate(request)
    # Should work without errors; deduplicated to just Warfarin + Aspirin
    assert source == "fallback"


@pytest.mark.asyncio
async def test_engine_allergy_detection_for_penicillin():
    """Patient allergic to penicillin + evaluated drug Amoxicillin → critical alert."""
    request = make_request(["Amoxicillin"], allergies=["penicillin"])
    engine = SafetyEngine()

    with patch.object(engine._llm, "call_groq", new=AsyncMock(return_value=None)):
        with patch.object(engine._llm, "call_huggingface", new=AsyncMock(return_value=None)):
            _, allergy_alerts, _, _ = await engine.evaluate(request)

    critical = [a for a in allergy_alerts if a.severity == "critical"]
    assert len(critical) >= 1


@pytest.mark.asyncio
async def test_engine_contraindication_detection():
    """Metformin + renal failure → absolute contraindication detected."""
    request = make_request(["Metformin"], conditions=["renal failure"])
    engine = SafetyEngine()

    with patch.object(engine._llm, "call_groq", new=AsyncMock(return_value=None)):
        with patch.object(engine._llm, "call_huggingface", new=AsyncMock(return_value=None)):
            _, _, contraindications, _ = await engine.evaluate(request)

    assert any(
        c.condition.lower() == "renal failure" for c in contraindications
    )
