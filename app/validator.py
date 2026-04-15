"""
EvoDoc — Clinical Drug Safety Engine
Input Validator & Normalizer

Handles drug name normalization, deduplication, misspelling correction,
and pre-processing before the engine pipeline runs.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Misspelling correction dictionary (common clinical drug name errors)
# ---------------------------------------------------------------------------

DRUG_CORRECTIONS: dict[str, str] = {
    # Warfarin variants
    "warferin": "Warfarin",
    "warfarrin": "Warfarin",
    "coumadin": "Warfarin",
    # Aspirin
    "aspirn": "Aspirin",
    "acetylsalicylic acid": "Aspirin",
    "asa": "Aspirin",
    # Metformin
    "metfromin": "Metformin",
    "glucophage": "Metformin",
    # Ibuprofen
    "iboprofen": "Ibuprofen",
    "ibuprofin": "Ibuprofen",
    "advil": "Ibuprofen",
    "motrin": "Ibuprofen",
    # Lisinopril
    "lisinpril": "Lisinopril",
    "lisopril": "Lisinopril",
    # Simvastatin
    "simavstatin": "Simvastatin",
    "zocor": "Simvastatin",
    # Amoxicillin
    "amoxycillin": "Amoxicillin",
    "amoxil": "Amoxicillin",
    # Metoprolol
    "metropolol": "Metoprolol",
    "lopressor": "Metoprolol",
    # Omeprazole
    "omeprazol": "Omeprazole",
    "prilosec": "Omeprazole",
    # Clarithromycin
    "clarithromiycin": "Clarithromycin",
    "biaxin": "Clarithromycin",
    # Ciprofloxacin
    "cipro": "Ciprofloxacin",
    "ciproflox": "Ciprofloxacin",
    # Sildenafil
    "viagra": "Sildenafil",
    "sildanafil": "Sildenafil",
    # Fluoxetine
    "prozac": "Fluoxetine",
    "fluoxetene": "Fluoxetine",
    # Clopidogrel
    "plavix": "Clopidogrel",
    "clopidogral": "Clopidogrel",
    # Digoxin
    "lanoxin": "Digoxin",
    "digoxen": "Digoxin",
    # Amiodarone
    "codarone": "Amiodarone",
    "amiodaronne": "Amiodarone",
    # Lithium
    "lithum": "Lithium",
    # Tramadol
    "ultram": "Tramadol",
    "tramdol": "Tramadol",
    # Phenytoin
    "dilantin": "Phenytoin",
    "phenytoim": "Phenytoin",
    # Rifampicin
    "rifampin": "Rifampicin",
    "rifampcin": "Rifampicin",
    # Theophylline
    "theofylline": "Theophylline",
    "theophyline": "Theophylline",
    # Nitroglycerin/nitrates
    "nitroglycerin": "Nitrates",
    "ntg": "Nitrates",
    "isosorbide": "Nitrates",
    # Verapamil
    "verapimil": "Verapamil",
    "calan": "Verapamil",
}

# ---------------------------------------------------------------------------
# Drug class normalization — map brand/class aliases to canonical names
# ---------------------------------------------------------------------------

DRUG_CLASS_ALIASES: dict[str, str] = {
    "ace inhibitor": "ACE Inhibitors",
    "ace inhibitors": "ACE Inhibitors",
    "arb": "ARBs",
    "arbs": "ARBs",
    "angiotensin receptor blocker": "ARBs",
    "ssri": "SSRIs",
    "ssris": "SSRIs",
    "selective serotonin reuptake inhibitor": "SSRIs",
    "maoi": "MAOIs",
    "monoamine oxidase inhibitor": "MAOIs",
    "nsaid": "NSAIDs",
    "nsaids": "NSAIDs",
    "non-steroidal anti-inflammatory": "NSAIDs",
    "oral contraceptive": "Oral Contraceptives",
    "ocp": "Oral Contraceptives",
    "birth control": "Oral Contraceptives",
    "fluoroquinolone": "Fluoroquinolones",
    "antacid": "Antacids",
    "calcium channel blocker": "Verapamil",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_drug_name(name: str) -> str:
    """
    Apply correction dict, then class alias dict, then title-case.
    Returns a canonical drug name string.
    """
    cleaned = name.strip().lower()
    # Remove non-alphanumeric except spaces and hyphens
    cleaned = re.sub(r"[^a-z0-9\s\-]", "", cleaned)

    # Check misspelling corrections first (exact match after cleaning)
    if cleaned in DRUG_CORRECTIONS:
        return DRUG_CORRECTIONS[cleaned]

    # Check drug class aliases
    if cleaned in DRUG_CLASS_ALIASES:
        return DRUG_CLASS_ALIASES[cleaned]

    # Fall back to title-case
    return name.strip().title()


def normalize_drug_list(drugs: list[str]) -> list[str]:
    """
    Normalize, correct, and deduplicate a list of drug names.
    Preserves order of first occurrence.
    """
    seen: set[str] = set()
    result: list[str] = []
    for raw in drugs:
        if not raw or not raw.strip():
            continue
        normalized = normalize_drug_name(raw)
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def normalize_condition_list(conditions: list[str]) -> list[str]:
    """Lowercase and strip condition names, deduplicate."""
    seen: set[str] = set()
    result: list[str] = []
    for c in conditions:
        if not c or not c.strip():
            continue
        normalized = c.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def normalize_allergy_list(allergies: list[str]) -> list[str]:
    """Lowercase and strip allergy names, deduplicate."""
    return normalize_condition_list(allergies)
