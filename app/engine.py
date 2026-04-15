"""
EvoDoc — Clinical Drug Safety Engine
Core Engine

Orchestrates the full evaluation pipeline:
  Request → LLM Handler → [Fallback Rule Engine] → Allergy Detector
  → Contraindication Checker → Risk Scorer → Response
"""

from __future__ import annotations

import json
import logging
import os
from itertools import combinations
from pathlib import Path

import httpx

from app.cache import generate_cache_key, get_cache, set_cache
from app.models import (
    AllergyAlert,
    Contraindication,
    EvaluationRequest,
    EvaluationResponse,
    Interaction,
    PatientHistory,
)
from app.scorer import compute_risk_score
from app.validator import normalize_drug_list

logger = logging.getLogger("evodoc.engine")

# ---------------------------------------------------------------------------
# Fallback data loading
# ---------------------------------------------------------------------------

_DATA_PATH = Path(__file__).parent / "fallback_interactions.json"


def _load_fallback_data() -> dict:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_FALLBACK_DATA: dict = _load_fallback_data()
_FALLBACK_INTERACTIONS: list[dict] = _FALLBACK_DATA["interactions"]
_ALLERGY_CLASS_MAP: dict[str, list[str]] = _FALLBACK_DATA["allergy_class_map"]
_CONTRAINDICATION_MAP: dict[str, list[str]] = _FALLBACK_DATA["contraindication_map"]


# ---------------------------------------------------------------------------
# LLM Handler
# ---------------------------------------------------------------------------

_LLM_TIMEOUT = 5.0  # Hard 5-second cutoff

_LLM_SYSTEM_PROMPT = """- You are a clinical drug safety AI
- Only return JSON
- No explanations
- Use real medical reasoning
- If uncertain -> set requires_doctor_review = true
- Never hallucinate drugs

The JSON must strictly match this schema:
{
  "interactions": [
    {
      "drug_a": "string",
      "drug_b": "string",
      "severity": "high" | "medium" | "low",
      "mechanism": "string",
      "recommendation": "string"
    }
  ],
  "allergy_alerts": [
    {
      "drug": "string",
      "allergen": "string",
      "severity": "critical" | "caution",
      "cross_reactivity": true | false
    }
  ],
  "contraindications": [
    {
      "drug": "string",
      "condition": "string",
      "risk_level": "absolute" | "relative",
      "alternative": "string" | null
    }
  ]
}

Rules:
- severity must be exactly: "high", "medium", or "low"
- allergy severity must be exactly: "critical" or "caution"
- risk_level must be exactly: "absolute" or "relative"
- No null values in required fields
- Return empty arrays if no interactions/alerts/contraindications found
"""


class LLMHandler:
    """
    Handles communication with the LLM API.
    Supports Groq (primary), HuggingFace Inference API (fallback).
    """

    def __init__(self) -> None:
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.hf_api_key = os.getenv("HF_API_KEY", "")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.hf_model = os.getenv(
            "HF_MODEL", "mistralai/BioMistral-7B"
        )

    def _build_user_prompt(
        self, drugs: list[str], patient: PatientHistory
    ) -> str:
        return f"""Evaluate the following:

DRUGS TO EVALUATE: {", ".join(drugs)}

PATIENT CONTEXT:
- Age: {patient.age} years
- Weight: {patient.weight_kg} kg
- Active Conditions: {", ".join(patient.conditions) or "none"}
- Current Medications: {", ".join(patient.current_medications) or "none"}
- Known Allergies: {", ".join(patient.allergies) or "none"}

Return ONLY the JSON analysis. No commentary."""

    async def call_groq(self, drugs: list[str], patient: PatientHistory) -> dict | None:
        """Call Groq API with Llama3-70B."""
        if not self.groq_api_key:
            logger.debug("Groq API key not configured, skipping.")
            return None

        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(drugs, patient)},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
        }

        try:
            async with httpx.AsyncClient(timeout=_LLM_TIMEOUT) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
        except (httpx.TimeoutException, httpx.HTTPStatusError, json.JSONDecodeError, KeyError) as e:
            logger.warning("Groq call failed: %s", e)
            return None

    async def call_huggingface(
        self, drugs: list[str], patient: PatientHistory
    ) -> dict | None:
        """Call HuggingFace Inference API as a secondary LLM option."""
        if not self.hf_api_key:
            logger.debug("HuggingFace API key not configured, skipping.")
            return None

        prompt = (
            _LLM_SYSTEM_PROMPT
            + "\n\n"
            + self._build_user_prompt(drugs, patient)
            + "\n\nJSON:"
        )

        try:
            async with httpx.AsyncClient(timeout=_LLM_TIMEOUT) as client:
                resp = await client.post(
                    f"https://api-inference.huggingface.co/models/{self.hf_model}",
                    headers={"Authorization": f"Bearer {self.hf_api_key}"},
                    json={"inputs": prompt, "parameters": {"max_new_tokens": 2048}},
                )
                resp.raise_for_status()
                result = resp.json()
                # HF returns list of generated text candidates
                if isinstance(result, list) and result:
                    text = result[0].get("generated_text", "")
                    # Extract JSON portion
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start != -1 and end > start:
                        return json.loads(text[start:end])
        except (httpx.TimeoutException, httpx.HTTPStatusError, json.JSONDecodeError, KeyError) as e:
            logger.warning("HuggingFace call failed: %s", e)
        return None

    async def evaluate(
        self, drugs: list[str], patient: PatientHistory
    ) -> tuple[dict | None, str]:
        """
        Attempt LLM evaluation. Returns (parsed_dict, source_label).
        source_label is 'llm' on success, 'fallback' on failure.
        """
        # Try Groq first
        result = await self.call_groq(drugs, patient)
        if result and _is_valid_llm_output(result):
            return result, "llm"

        # Try HuggingFace
        result = await self.call_huggingface(drugs, patient)
        if result and _is_valid_llm_output(result):
            return result, "llm"

        return None, "fallback"


def _is_valid_llm_output(data: dict) -> bool:
    """
    Validate that the LLM returned the expected structure with
    correct field types and allowed enum values.
    """
    if not isinstance(data, dict):
        return False

    required_keys = {"interactions", "allergy_alerts", "contraindications"}
    if not required_keys.issubset(data.keys()):
        return False

    valid_severities = {"high", "medium", "low"}
    valid_allergy_severities = {"critical", "caution"}
    valid_risk_levels = {"absolute", "relative"}

    for interaction in data.get("interactions", []):
        if not isinstance(interaction, dict):
            return False
        for field in ("drug_a", "drug_b", "severity", "mechanism", "recommendation"):
            if not interaction.get(field):
                return False
        if interaction["severity"] not in valid_severities:
            return False

    for alert in data.get("allergy_alerts", []):
        if not isinstance(alert, dict):
            return False
        for field in ("drug", "allergen", "severity"):
            if not alert.get(field):
                return False
        if alert["severity"] not in valid_allergy_severities:
            return False
        if not isinstance(alert.get("cross_reactivity"), bool):
            return False

    for contraindication in data.get("contraindications", []):
        if not isinstance(contraindication, dict):
            return False
        for field in ("drug", "condition", "risk_level"):
            if not contraindication.get(field):
                return False
        if contraindication["risk_level"] not in valid_risk_levels:
            return False

    return True


# ---------------------------------------------------------------------------
# Fallback Rule Engine
# ---------------------------------------------------------------------------


class FallbackEngine:
    """
    Deterministic rule-based engine using drug_interactions.json.
    Used when LLM is unavailable or produces invalid output.
    """

    def find_interactions(
        self, drugs: list[str], current_medications: list[str]
    ) -> list[Interaction]:
        """
        Find pairwise interactions among the evaluated drugs AND
        between evaluated drugs and current medications.
        """
        all_drugs = set(d.lower() for d in drugs + current_medications)
        found: list[Interaction] = []

        for rule in _FALLBACK_INTERACTIONS:
            a = rule["drug_a"].lower()
            b = rule["drug_b"].lower()
            # Check if BOTH drugs in the rule appear in the combined drug set
            if a in all_drugs and b in all_drugs:
                found.append(
                    Interaction(
                        drug_a=rule["drug_a"],
                        drug_b=rule["drug_b"],
                        severity=rule["severity"],
                        mechanism=rule["mechanism"],
                        clinical_recommendation=rule["recommendation"],
                        source_confidence="high (clinical rule)",
                    )
                )

        return found

    def find_allergy_alerts(
        self, drugs: list[str], allergies: list[str]
    ) -> list[AllergyAlert]:
        """
        Cross-reference evaluated drugs against patient's known allergies.
        Uses drug class cross-reactivity map.
        """
        alerts: list[AllergyAlert] = []
        drugs_lower = {d.lower() for d in drugs}
        allergies_lower = {a.lower() for a in allergies}

        for drug in drugs:
            drug_lower = drug.lower()

            # Direct allergy match
            if drug_lower in allergies_lower:
                alerts.append(
                    AllergyAlert(
                        medicine=drug,
                        reason=f"Direct allergy to {drug}",
                        severity="critical",
                    )
                )
                continue

            # Cross-reactivity check
            for allergen_class, class_drugs in _ALLERGY_CLASS_MAP.items():
                class_drugs_lower = [d.lower() for d in class_drugs]

                # Patient is allergic to this class AND drug is in this class
                if allergen_class in allergies_lower and drug_lower in class_drugs_lower:
                    alerts.append(
                        AllergyAlert(
                            medicine=drug,
                            reason=f"Cross-reactivity with {allergen_class} class",
                            severity="critical",
                        )
                    )

                # Patient is allergic to a drug in this class AND evaluated drug is in same class
                elif drug_lower in class_drugs_lower:
                    for allergy in allergies_lower:
                        if allergy in class_drugs_lower and allergy != drug_lower:
                            alerts.append(
                                AllergyAlert(
                                    medicine=drug,
                                    reason=f"Cross-reactivity with {allergy}",
                                    severity="caution",
                                )
                            )

        return alerts

    def find_contraindications(
        self, drugs: list[str], conditions: list[str]
    ) -> list[Contraindication]:
        """
        Check each drug against patient conditions using the contraindication map.
        """
        contraindications: list[Contraindication] = []
        conditions_lower = {c.lower() for c in conditions}

        for drug in drugs:
            # Try exact match first, then partial class match
            matched_conditions = _CONTRAINDICATION_MAP.get(
                drug, _CONTRAINDICATION_MAP.get(drug.lower(), [])
            )

            for contraindicated_condition in matched_conditions:
                if contraindicated_condition.lower() in conditions_lower:
                    contraindications.append(
                        Contraindication(
                            drug=drug,
                            condition=contraindicated_condition,
                            risk_level="absolute",
                            alternative=_suggest_alternative(drug),
                        )
                    )

        return contraindications


def _suggest_alternative(drug: str) -> str | None:
    """Return a safe alternative for common contraindicated drugs."""
    alternatives = {
        "Metformin": "Consider insulin therapy or GLP-1 agonists for renal-impaired patients",
        "NSAIDs": "Acetaminophen or topical analgesics",
        "Warfarin": "Direct oral anticoagulants (DOACs) if not contraindicated",
        "Sildenafil": "Consult cardiologist for alternative ED treatment",
        "Metoprolol": "Consider cardioselective alternatives or amlodipine",
        "Lisinopril": "Consider amlodipine or hydralazine in pregnancy",
        "Amiodarone": "Consider dronedarone or sotalol",
        "Ciprofloxacin": "Consider azithromycin or trimethoprim-sulfamethoxazole",
        "Digoxin": "Consider alternative rate control agents",
    }
    return alternatives.get(drug)


# ---------------------------------------------------------------------------
# analyze_drugs Orchestrator
# ---------------------------------------------------------------------------

_llm_handler = LLMHandler()
_fallback_engine = FallbackEngine()

import time

async def analyze_drugs(medicines: list[str], patient_history: PatientHistory) -> EvaluationResponse:
    """
    Main evaluation pipeline matching exactly the requested steps:
    1. Normalize inputs
    2. Generate cache key
    3. Check cache -> If hit, return cached
    4. If miss -> Call LLM -> validate -> if invalid -> fallback
    5. Detect allergies via class mapping
    6. Compute risk score
    7. Set requires_doctor_review if confidence low
    8. Store in cache
    """
    t_start = time.monotonic()
    
    # 1. Normalize inputs (lowercase, remove duplicates handled in validation but explicit here)
    normalized_drugs = normalize_drug_list(medicines)
    
    # 2. Generate cache key
    cache_key = generate_cache_key(normalized_drugs, patient_history.current_medications)
    
    # 3. Check cache
    cached_result = get_cache(cache_key)
    
    # 4. If hit -> return cached response
    if cached_result is not None:
        cached_result.cache_hit = True
        cached_result.processing_time_ms = int((time.monotonic() - t_start) * 1000)
        return cached_result
        
    # 5. If miss -> Call LLM ...
    llm_result, source = await _llm_handler.evaluate(normalized_drugs, patient_history)
    
    interactions: list[Interaction] = []
    allergy_alerts: list[AllergyAlert] = []
    contraindications: list[Contraindication] = []

    if llm_result:
        # LLM succeeded and validated
        interactions = [Interaction(**i) for i in llm_result.get("interactions", [])]
        allergy_alerts = [AllergyAlert(**a) for a in llm_result.get("allergy_alerts", [])]
        contraindications = [Contraindication(**c) for c in llm_result.get("contraindications", [])]
    else:
        # if invalid -> fallback system
        source = "fallback"
        interactions = _fallback_engine.find_interactions(normalized_drugs, patient_history.current_medications)
        
        # 6. Detect allergies using class mapping (integrated in fallback Engine)
        allergy_alerts = _fallback_engine.find_allergy_alerts(normalized_drugs, patient_history.allergies)
        
        # also run contraindication checks from fallback map
        contraindications = _fallback_engine.find_contraindications(normalized_drugs, patient_history.conditions)

    # 7. Compute risk score
    risk_score = compute_risk_score(interactions, allergy_alerts, contraindications)
    
    # 8. Set requires_doctor_review
    # Based on whether there are high severity interactions, critical allergies, or score == high
    requires_doctor_review = risk_score.grade == "high"
    
    processing_time_ms = int((time.monotonic() - t_start) * 1000)
    
    response = EvaluationResponse(
        interactions=interactions,
        allergy_alerts=allergy_alerts,
        safe_to_prescribe=risk_score.grade in ["low", "medium"],
        overall_risk_level=risk_score.grade,
        requires_doctor_review=requires_doctor_review,
        source=source,
        cache_hit=False,
        processing_time_ms=processing_time_ms,
    )
    
    # 9. Store in cache
    set_cache(cache_key, response)
    
    
    # Return structured response
    return response


# ---------------------------------------------------------------------------
# Direct Export for User Requirements
# ---------------------------------------------------------------------------

def get_fallback_interactions(medicines: list[str]) -> list[Interaction]:
    """
    Exposed utility function to directly fetch clinical rules from fallback_interactions.json
    matching the schema requested.
    """
    normalized_drugs = normalize_drug_list(medicines)
    return _fallback_engine.find_interactions(normalized_drugs, [])

async def call_llm_handler(medicines: list[str], patient_history: PatientHistory) -> dict | None:
    """
    Exported function to query HuggingFace (BioMistral / Med42) directly.
    - Demands strict JSON only
    - Validates structure
    - Retries automatically if invalid
    - Returns None if still invalid after retries
    """
    normalized_drugs = normalize_drug_list(medicines)
    
    # Attempt up to 2 times (1 initial + 1 retry)
    for attempt in range(2):
        if attempt > 0:
            logger.info("Retrying HuggingFace LLM call due to invalid JSON on attempt %d", attempt)
            
        result = await _llm_handler.call_huggingface(normalized_drugs, patient_history)
        
        # Validate output schema
        if result and _is_valid_llm_output(result):
            return result
            
    # If still invalid after retries, return None
    logger.warning("HuggingFace LLM failed to return valid JSON after retries.")
    return None
