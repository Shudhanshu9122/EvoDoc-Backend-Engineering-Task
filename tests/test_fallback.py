"""
Tests for all 17 fallback interaction rules, allergy cross-reactivity,
and contraindication detection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app.engine import FallbackEngine


@pytest.fixture
def engine():
    return FallbackEngine()


# ---------------------------------------------------------------------------
# Interaction tests — all 17 rules
# ---------------------------------------------------------------------------


INTERACTION_RULES = [
    ("Warfarin", "Aspirin", "high"),
    ("Metformin", "Alcohol", "high"),
    ("Simvastatin", "Clarithromycin", "high"),
    ("Lisinopril", "Potassium", "high"),
    ("Fluoxetine", "MAOIs", "high"),
    ("Ibuprofen", "Warfarin", "medium"),
    ("Metoprolol", "Verapamil", "high"),
    ("Ciprofloxacin", "Theophylline", "high"),
    ("Digoxin", "Amiodarone", "high"),
    ("Clopidogrel", "Omeprazole", "medium"),
    ("Lithium", "NSAIDs", "high"),
    ("Sildenafil", "Nitrates", "high"),
    ("Tramadol", "SSRIs", "medium"),
    ("Phenytoin", "Warfarin", "medium"),
    ("Rifampicin", "Oral Contraceptives", "medium"),
    ("ACE Inhibitors", "ARBs", "medium"),
    ("Fluoroquinolones", "Antacids", "low"),
]


@pytest.mark.parametrize("drug_a, drug_b, expected_severity", INTERACTION_RULES)
def test_fallback_interaction_rule(engine, drug_a, drug_b, expected_severity):
    """Each of the 17 rules should be found when both drugs are present."""
    interactions = engine.find_interactions([drug_a, drug_b], [])
    assert len(interactions) >= 1, f"Expected interaction between {drug_a} and {drug_b}"
    matched = [i for i in interactions if (
        (i.drug_a.lower() == drug_a.lower() and i.drug_b.lower() == drug_b.lower()) or
        (i.drug_a.lower() == drug_b.lower() and i.drug_b.lower() == drug_a.lower())
    )]
    assert matched, f"No interaction found between {drug_a} and {drug_b}"
    assert matched[0].severity == expected_severity


def test_no_interaction_for_safe_drugs(engine):
    """Drugs with no known interaction should return empty list."""
    interactions = engine.find_interactions(["Vitamin C", "Zinc"], [])
    assert interactions == []


def test_interaction_found_via_current_medications(engine):
    """Interaction between new drug and existing medication should be detected."""
    interactions = engine.find_interactions(["Warfarin"], ["Aspirin"])
    assert any(
        "warfarin" in i.drug_a.lower() or "warfarin" in i.drug_b.lower()
        for i in interactions
    )


def test_duplicate_drugs_not_double_counted(engine):
    """Duplicate drugs should not produce extra interactions."""
    interactions_single = engine.find_interactions(["Warfarin", "Aspirin"], [])
    interactions_double = engine.find_interactions(["Warfarin", "Aspirin", "Warfarin"], [])
    assert len(interactions_single) == len(interactions_double)


# ---------------------------------------------------------------------------
# Allergy tests
# ---------------------------------------------------------------------------


def test_direct_allergy_match(engine):
    """Patient directly allergic to the drug → critical alert."""
    alerts = engine.find_allergy_alerts(["Ibuprofen"], ["ibuprofen"])
    assert len(alerts) == 1
    assert alerts[0].severity == "critical"
    assert alerts[0].cross_reactivity is False


def test_penicillin_cross_reactivity_to_amoxicillin(engine):
    """Penicillin allergy → Amoxicillin should trigger critical cross-reactivity alert."""
    alerts = engine.find_allergy_alerts(["Amoxicillin"], ["penicillin"])
    critical_alerts = [a for a in alerts if a.severity == "critical"]
    assert len(critical_alerts) >= 1
    assert any(a.cross_reactivity for a in critical_alerts)


def test_nsaid_class_cross_reactivity(engine):
    """Patient allergic to ibuprofen → naproxen should trigger caution alert."""
    alerts = engine.find_allergy_alerts(["Naproxen"], ["ibuprofen"])
    caution_alerts = [a for a in alerts if a.severity == "caution"]
    assert len(caution_alerts) >= 1


def test_no_allergy_alert_for_unrelated_drug(engine):
    """No allergy alert when drug has no relation to known allergens."""
    alerts = engine.find_allergy_alerts(["Metformin"], ["penicillin"])
    assert alerts == []


def test_no_allergies_produces_empty_alerts(engine):
    """Empty allergy list → no alerts."""
    alerts = engine.find_allergy_alerts(["Warfarin", "Digoxin"], [])
    assert alerts == []


# ---------------------------------------------------------------------------
# Contraindication tests
# ---------------------------------------------------------------------------


def test_metformin_contraindicated_in_renal_failure(engine):
    contraindications = engine.find_contraindications(["Metformin"], ["renal failure"])
    assert len(contraindications) >= 1
    assert contraindications[0].risk_level == "absolute"


def test_nsaid_contraindicated_in_peptic_ulcer(engine):
    contraindications = engine.find_contraindications(["NSAIDs"], ["peptic ulcer"])
    assert len(contraindications) >= 1


def test_no_contraindication_for_safe_combination(engine):
    """Drug with no contraindication for present conditions returns empty list."""
    contraindications = engine.find_contraindications(["Vitamin D"], ["hypertension"])
    assert contraindications == []


def test_sildenafil_contraindicated_in_unstable_angina(engine):
    contraindications = engine.find_contraindications(["Sildenafil"], ["unstable angina"])
    assert len(contraindications) >= 1


def test_lisinopril_contraindicated_in_pregnancy(engine):
    contraindications = engine.find_contraindications(["Lisinopril"], ["pregnancy"])
    assert len(contraindications) >= 1
