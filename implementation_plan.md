# EvoDoc — Clinical Drug Safety Engine
## Architecture & Implementation Plan
**Author:** Senior Architect Review  
**Date:** 2026-04-15  
**Version:** 1.0  
**Status:** Approved for Implementation

---

## 1. Executive Summary

EvoDoc's Clinical Drug Safety Engine is a **production-grade FastAPI backend** designed to evaluate drug-drug interactions, detect allergy risks, apply patient-specific contraindications, and produce a normalized risk score — all within a **sub-3-second response time** SLA.

The system operates on a **LLM-first, rule-based fallback** architecture. The LLM provides rich clinical reasoning; a deterministic JSON rule engine guarantees correctness when the LLM is unavailable or produces invalid output. An in-memory cache layer prevents redundant computation.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          FastAPI Server (main.py)                    │
│  POST /api/v1/evaluate                                               │
│  GET  /api/v1/health                                                 │
│  GET  /api/v1/cache/stats                                            │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  Input Validator │  (models.py — Pydantic v2)
              │  - dedup drugs   │
              │  - normalize names│
              │  - validate types │
              └────────┬─────────┘
                       │
              ┌────────▼────────┐
              │   Cache Layer    │  (cache.py)
              │  TTL=3600s      │
              │  Key=SHA-256    │
              └──┬──────────┬───┘
               HIT         MISS
                │            │
                │   ┌────────▼──────────┐
                │   │  Safety Engine    │  (engine.py)
                │   │                  │
                │   │  ┌─────────────┐ │
                │   │  │ LLM Handler │ │  → Groq/BioMistral/Med42
                │   │  └──────┬──────┘ │
                │   │         │ fail   │
                │   │  ┌──────▼──────┐ │
                │   │  │  Fallback   │ │  → drug_interactions.json
                │   │  │  Rule Engine│ │
                │   │  └──────┬──────┘ │
                │   │         │        │
                │   │  ┌──────▼──────┐ │
                │   │  │  Allergy    │ │
                │   │  │  Detector   │ │
                │   │  └──────┬──────┘ │
                │   │         │        │
                │   │  ┌──────▼──────┐ │
                │   │  │   Risk      │ │
                │   │  │   Scorer    │ │
                │   │  └─────────────┘ │
                │   └────────┬─────────┘
                │            │
                └────────────▼
                         Response (JSON)
```

---

## 3. Component Breakdown

### 3.1 File Structure

```
evodocs/
├── main.py                     # FastAPI app, routes, middleware
├── engine.py                   # Core safety engine + LLM handler
├── cache.py                    # TTL cache implementation
├── models.py                   # Pydantic request/response models
├── validator.py                # Input normalization & dedup
├── scorer.py                   # Risk scoring algorithm
├── data/
│   └── drug_interactions.json  # 17 real-world fallback rules
├── tests/
│   ├── test_cache.py
│   ├── test_engine.py
│   ├── test_scorer.py
│   └── test_fallback.py
├── requirements.txt
└── .env.example
```

---

### 3.2 Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| API Server | `main.py` | Route handling, CORS, error middleware, request lifecycle |
| Safety Engine | `engine.py` | Orchestrates LLM → fallback → allergy → scoring pipeline |
| Cache | `cache.py` | SHA-256 keyed TTL cache, stats endpoint |
| Data Models | `models.py` | Pydantic v2 request/response schemas, field-level validators |
| Validator | `validator.py` | Drug name normalization, dedup, spell correction mapping |
| Risk Scorer | `scorer.py` | Weighted scoring, normalization to 0–100 |
| Fallback Data | `data/drug_interactions.json` | 17 real clinical interaction rules |

---

## 4. Data Models

### 4.1 Request Model

```python
class PatientContext(BaseModel):
    age: int                           # years
    weight_kg: float                   # kilograms
    conditions: list[str]              # e.g. ["diabetes", "hypertension"]
    current_medications: list[str]     # existing drug regimen
    allergies: list[str]               # known allergens / drug classes

class EvaluationRequest(BaseModel):
    drugs: list[str]                   # drugs to evaluate (≥1)
    patient: PatientContext
```

### 4.2 Response Model

```python
class Interaction(BaseModel):
    drug_a: str
    drug_b: str
    severity: Literal["high", "medium", "low"]
    mechanism: str
    recommendation: str

class AllergyAlert(BaseModel):
    drug: str
    allergen: str
    severity: Literal["critical", "caution"]
    cross_reactivity: bool

class Contraindication(BaseModel):
    drug: str
    condition: str
    risk_level: Literal["absolute", "relative"]
    alternative: str | None

class RiskScore(BaseModel):
    total: float                       # 0–100
    grade: Literal["SAFE", "CAUTION", "HIGH_RISK", "CRITICAL"]
    breakdown: dict[str, float]

class EvaluationResponse(BaseModel):
    request_id: str
    cached: bool
    evaluated_at: datetime
    source: Literal["llm", "fallback"]
    drugs_evaluated: list[str]
    interactions: list[Interaction]
    allergy_alerts: list[AllergyAlert]
    contraindications: list[Contraindication]
    risk_score: RiskScore
    processing_time_ms: float
```

---

## 5. Cache Strategy

### Key Generation
```
cache_key = SHA-256(
    sorted(drugs) + "|" + sorted(patient.current_medications)
)
```

> **Note:** Only drugs + current_medications feed the cache key. Age, weight, conditions, and allergies personalize the scoring stage downstream, not the interaction lookup.

### Implementation
- **Backend:** Python `dict` with timestamp metadata (zero external dependencies)
- **TTL:** 3600 seconds (1 hour)
- **Eviction:** Lazy eviction on access + periodic sweep every 5 minutes
- **Stats:** `/api/v1/cache/stats` → `{size, hits, misses, hit_rate}`

---

## 6. LLM Strategy

### Model Priority Order
1. **Groq API + Llama3-70B** — fastest, generous free tier
2. **BioMistral via HuggingFace Inference API**
3. **Med42 via HuggingFace**

### Strict JSON Enforcement
- System prompt enforces JSON-only response
- Output parsed with `json.loads()` — any parse error → fallback
- Required fields validated via Pydantic; missing/null fields → fallback
- Hallucinated severity values rejected (must be exactly high/medium/low)

### Prompt Template
```
You are a clinical pharmacology expert. Given the drug list and patient 
context below, return ONLY valid JSON matching this exact schema:
{"interactions": [...], "allergy_alerts": [...], "contraindications": [...]}

No explanation. No markdown fences. Pure JSON only.
```

---

## 7. Fallback Rule Engine — 17 Real Interactions

| # | Drug A | Drug B | Severity |
|---|--------|--------|----------|
| 1 | Warfarin | Aspirin | high |
| 2 | Metformin | Alcohol | high |
| 3 | Simvastatin | Clarithromycin | high |
| 4 | Lisinopril | Potassium | high |
| 5 | Fluoxetine | MAOIs | high |
| 6 | Ibuprofen | Warfarin | medium |
| 7 | Metoprolol | Verapamil | high |
| 8 | Ciprofloxacin | Theophylline | high |
| 9 | Digoxin | Amiodarone | high |
| 10 | Clopidogrel | Omeprazole | medium |
| 11 | Lithium | NSAIDs | high |
| 12 | Sildenafil | Nitrates | high |
| 13 | Tramadol | SSRIs | medium |
| 14 | Phenytoin | Warfarin | medium |
| 15 | Rifampicin | Oral Contraceptives | medium |
| 16 | ACE Inhibitors | ARBs | medium |
| 17 | Fluoroquinolones | Antacids | low |

### Drug Class → Allergy Cross-Reactivity Map
```json
{
  "penicillin": ["amoxicillin", "ampicillin", "dicloxacillin"],
  "sulfa": ["sulfamethoxazole", "trimethoprim-sulfamethoxazole"],
  "nsaid": ["ibuprofen", "naproxen", "aspirin", "diclofenac"],
  "cephalosporin": ["cephalexin", "cefazolin", "ceftriaxone"],
  "macrolide": ["azithromycin", "clarithromycin", "erythromycin"]
}
```

---

## 8. Risk Scoring Algorithm

### Accumulation
```
score = 0
HIGH interaction:     +30
MEDIUM interaction:   +15
LOW interaction:      +5
CRITICAL allergy:     +40
CAUTION allergy:      +20
ABSOLUTE contraind.:  +25
RELATIVE contraind.:  +10
```

### Grade Thresholds
| Score Range | Grade |
|-------------|-------|
| 0–20 | SAFE |
| 21–50 | CAUTION |
| 51–80 | HIGH_RISK |
| 81–100 | CRITICAL |

---

## 9. Validation Layer

| Check | Action |
|-------|--------|
| Duplicate drugs | Remove (case-insensitive) |
| Drug name casing | Normalize to Title Case |
| Known misspellings | Map via correction dict |
| Empty drugs list | 422 Unprocessable Entity |
| Null LLM fields | Reject → fallback |
| Invalid severity | Reject → fallback |
| Age < 0 or > 130 | 400 Bad Request |

---

## 10. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/evaluate` | POST | Main safety evaluation |
| `/api/v1/health` | GET | Service liveness + uptime |
| `/api/v1/cache/stats` | GET | Cache performance metrics |
| `/api/v1/cache/flush` | DELETE | Admin cache clear |

---

## 11. Testing Strategy

| Test File | Coverage |
|-----------|----------|
| `test_cache.py` | TTL expiry, hit/miss, stats, flush |
| `test_engine.py` | Full pipeline, LLM path, fallback trigger |
| `test_scorer.py` | Score calc, normalization, grade assignment |
| `test_fallback.py` | All 17 rules correctly triggered |

---

## 12. Non-Functional Requirements

| NFR | Target |
|-----|--------|
| Response time | < 3s p95 |
| LLM timeout | 5s hard cutoff → fallback |
| Cache TTL | 3600s |
| Min fallback rules | 17 |
| Python version | 3.11+ |

---

## 13. Delivery Phases

| Phase | Scope |
|-------|-------|
| 1 | models.py, validator.py, scorer.py, fallback JSON |
| 2 | cache.py, engine.py (LLM + fallback pipeline) |
| 3 | main.py (FastAPI routes + middleware) |
| 4 | Full test suite |
| 5 | requirements.txt, .env.example, README |

---

*End of Architecture Plan — EvoDoc Clinical Drug Safety Engine v1.0*
