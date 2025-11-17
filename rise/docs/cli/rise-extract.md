# rise-extract: Entity and Event Extraction

## Mission

Extract structured intelligence - entities (military units, locations, persons, organizations), events (attacks, arrests, movements), and relationships - from normalized evidence with **explicit confidence scoring** and **complete traceability**.

## Core Philosophy

You are a **forensic intelligence analyst** identifying who, what, when, where, and how from messy field reports. Your extraction must:
- Never invent information (low confidence > hallucination)
- Preserve original Burmese terminology alongside translations
- Score confidence for every extraction (0.0-1.0)
- Document extraction method (rule-based, ML, human-verified)
- Flag ambiguities explicitly
- Cross-reference against context databases (military units, townships)
- Create audit trail showing WHY you extracted what you did

**Rule: Uncertain extraction with documented uncertainty > confident hallucination**

---

## Usage

### Basic Entity Extraction
```bash
rise-extract \
  --input evidence_bundles/bundles/ \
  --entities military_units,locations,persons \
  --context myanmar_sagaing \
  --output extracted_entities/
```

### Event-Focused Extraction
```bash
rise-extract \
  --input evidence_bundles/bundles/ \
  --events attacks,arrests,movements \
  --temporal-extraction \
  --context myanmar \
  --confidence-threshold 0.7 \
  --output extracted_events/
```

### Full Extraction with Relationship Mapping
```bash
rise-extract \
  --input evidence_bundles/bundles/ \
  --entities all \
  --events all \
  --relationships command_chains,victim_perpetrator \
  --context myanmar_tatmadaw_2024 \
  --output extracted_full/ \
  --audit-trail extraction_decisions.log
```

---

## Parameters

### Input Parameters

**`--input <path>`** (required)
- Directory of evidence bundles (from rise-ingest)
- Can also be single evidence bundle JSON
- Expects normalized evidence with Unicode text

**`--context <name>`** (required)
- Context library: `myanmar`, `myanmar_sagaing`, `myanmar_tatmadaw_2024`
- Loads:
  - Military unit databases
  - Township/village lists
  - Ethnic group terminology
  - Known persons/organizations
  - Burmese-English term mappings

### Extraction Targets

**`--entities <types>`** (optional, default: all)
- Comma-separated: `military_units`, `locations`, `persons`, `organizations`, `weapons`, `vehicles`
- Special: `all` extracts everything

**`--events <types>`** (optional, default: all)
- Comma-separated: `attacks`, `arrests`, `detentions`, `displacements`, `killings`, `torture`, `movements`, `meetings`
- Special: `all` extracts everything

**`--relationships <types>`** (optional)
- Comma-separated: `command_chains`, `alliances`, `victim_perpetrator`, `witness_event`, `family`
- Builds entity graphs

**`--temporal-extraction`** (optional)
- Extract dates, times, durations
- Normalize Myanmar calendar → Gregorian
- Handle fuzzy temporal markers ("last week", "during harvest season")

**`--spatial-extraction`** (optional)
- Extract coordinates, place names, relative locations
- Geocode against township database
- Handle nested locations (village → township → region)

### Confidence & Quality

**`--confidence-threshold <float>`** (optional, default: 0.5)
- Minimum confidence (0.0-1.0) to include extraction
- Below threshold → still extracted but flagged as "low_confidence"

**`--require-context-match`** (optional)
- Only extract entities that match context database
- Prevents "hallucinated" military units
- Trade-off: might miss newly formed units

**`--preserve-burmese`** (optional, default: true)
- Keep original Burmese terms alongside English translations
- Critical for verification and re-analysis

### Processing

**`--method <approach>`** (optional, default: hybrid)
- `rule_based`: Pattern matching and regex
- `ml`: Neural NER models
- `hybrid`: Both, with consensus scoring
- `human_in_loop`: Interactive extraction with review

**`--parallel-workers <N>`** (optional)
- Number of parallel processing workers
- Default: CPU count / 2

### Output Parameters

**`--output <path>`** (required)
- Directory for extracted entities/events
- Structure: `<output>/entities/`, `<output>/events/`, `<output>/relationships/`

**`--audit-trail <path>`** (optional)
- Log every extraction decision
- Format: evidence_id, text_span, entity_extracted, confidence, method, reasoning

**`--review-queue <path>`** (optional)
- Output low-confidence extractions for human review
- Separate directory for items needing verification

---

## Output Structure

### Extracted Entity Format

```json
{
  "entity_id": "ent_mil_001",
  "entity_type": "military_unit",
  "extraction_timestamp": "2024-01-10T15:30:00Z",
  "extraction_tool": "rise-extract v1.2.0",

  "entity_data": {
    "name_burmese": "လိုင်း ၉၉",
    "name_english": "LID 99",
    "full_name": "Light Infantry Division 99",
    "aliases": ["99th Division", "Infantry Division 99"],

    "unit_type": "light_infantry_division",
    "parent_unit": {
      "entity_id": "ent_mil_parent_001",
      "name": "Northern Command"
    },
    "subordinate_units": [
      {"entity_id": "ent_mil_sub_001", "name": "Battalion 1"},
      {"entity_id": "ent_mil_sub_002", "name": "Battalion 2"}
    ],

    "attributes": {
      "commander": "Unknown",
      "base_location": "Kale_Township",
      "active_period": "2021-02-01 to present"
    }
  },

  "extraction_provenance": {
    "source_evidence": [
      {
        "evidence_id": "evd_042",
        "text_span": "လိုင်း ၉၉ အမှတ် ၁၄:၀၀ နာရီတွင် ရွာကို ပစ်ခတ်ခဲ့သည်",
        "character_offsets": [0, 8],
        "extraction_method": "ml_ner_burmese_v2.0",
        "confidence": 0.95,
        "reasoning": "Pattern matches LID naming convention, confirmed against military unit database"
      },
      {
        "evidence_id": "evd_118",
        "text_span": "99th Division shelled the village",
        "character_offsets": [0, 13],
        "extraction_method": "rule_based_english",
        "confidence": 0.92,
        "reasoning": "English variant, corroborates Burmese extraction"
      }
    ],

    "context_validation": {
      "matched_database_entry": true,
      "database_id": "tatmadaw_lid_99",
      "database_confidence": 1.0,
      "known_aliases_matched": ["LID 99", "Light Infantry Division 99"]
    },

    "confidence_breakdown": {
      "overall_confidence": 0.95,
      "extraction_confidence": 0.95,
      "context_match_confidence": 1.0,
      "multi_source_boost": 0.05  // Two sources corroborate
    }
  },

  "relationships": [
    {
      "relationship_type": "command_chain",
      "related_entity": "ent_mil_parent_001",
      "direction": "subordinate_to"
    },
    {
      "relationship_type": "perpetrator_of",
      "related_event": "evt_attack_001",
      "confidence": 0.9
    }
  ],

  "quality_flags": {
    "multiple_sources": true,
    "context_validated": true,
    "human_reviewed": false,
    "ambiguity_noted": false,
    "requires_verification": false
  }
}
```

### Extracted Event Format

```json
{
  "event_id": "evt_attack_001",
  "event_type": "artillery_attack",
  "extraction_timestamp": "2024-01-10T15:35:00Z",

  "event_data": {
    "description": "Artillery attack on Thayetchaung Village by LID 99",

    "temporal": {
      "date": "2021-04-15",
      "time": "14:00",
      "timezone": "Asia/Yangon",
      "date_precision": "exact",
      "time_precision": "approximate_hour",
      "original_text": "၁၄:၀၀ နာရီတွင်",
      "myanmar_calendar_date": "1382_Kason_15",  // Burmese calendar
      "extraction_confidence": 0.88
    },

    "spatial": {
      "primary_location": {
        "entity_id": "ent_loc_001",
        "name_burmese": "သရက်ချောင်းရွာ",
        "name_english": "Thayetchaung Village",
        "township": "Kale_Township",
        "region": "Sagaing_Region",
        "coordinates": {"lat": 23.2, "lon": 94.1, "precision": "village"},
        "extraction_confidence": 0.92
      },
      "affected_area_radius_km": 2
    },

    "entities_involved": {
      "perpetrators": [
        {
          "entity_id": "ent_mil_001",
          "role": "attacking_force",
          "confidence": 0.95,
          "evidence": "evd_042, evd_118"
        }
      ],
      "victims": [
        {
          "entity_id": "ent_per_001",
          "role": "injured_civilian",
          "name": "Ko Aung",
          "confidence": 0.9,
          "evidence": "evd_042"
        }
      ],
      "witnesses": [
        {
          "entity_id": "ent_per_002",
          "role": "eyewitness",
          "confidence": 0.85,
          "evidence": "evd_119"
        }
      ]
    },

    "incident_details": {
      "weapon_type": "artillery",
      "estimated_rounds": "15-20",
      "casualties": {
        "killed": 0,
        "injured": 1,
        "confidence": 0.8,
        "note": "Based on available evidence, actual numbers may be higher"
      },
      "damage": [
        {"type": "residential_building", "count": 3, "severity": "destroyed"},
        {"type": "monastery", "count": 1, "severity": "damaged"}
      ]
    }
  },

  "extraction_provenance": {
    "source_evidence": [
      {
        "evidence_id": "evd_042",
        "relevance": "primary_description",
        "confidence": 0.95
      }
    ],
    "extraction_method": "hybrid_ml_rules",
    "confidence_breakdown": {
      "overall": 0.88,
      "temporal": 0.88,
      "spatial": 0.92,
      "entities": 0.90,
      "incident_type": 0.95
    }
  },

  "verification_notes": [
    {
      "type": "single_source_claim",
      "note": "Casualty count from single source, requires corroboration",
      "priority": "high"
    }
  ]
}
```

---

## Extraction Methods

### Rule-Based Extraction
- Regex patterns for known entities
- Myanmar military unit patterns: `LID \d+`, `Infantry Battalion \d+`
- Date patterns: Gregorian + Myanmar calendar
- Location patterns: Township/village name lists

**Pros**: High precision, auditable, fast
**Cons**: Requires manual rule creation, misses variations

### ML-Based Extraction
- Neural NER models (fine-tuned on Myanmar conflict data)
- Contextual embeddings (BERT/RoBERTa)
- Character-level models (handle Burmese script)

**Pros**: Handles variations, learns from data
**Cons**: Requires training data, harder to audit, potential hallucination

### Hybrid Approach (Recommended)
```
Text Span
  ↓
Rule-based extraction → Candidates A (confidence via pattern match)
  ↓
ML extraction → Candidates B (confidence via model score)
  ↓
Consensus:
  - If both agree → boost confidence
  - If disagree → lower confidence, flag for review
  - If only one → use that one, note single-method
  ↓
Context validation → Check against database
  ↓
Final extraction with confidence score
```

---

## Context Database Matching

### Military Unit Database
```json
{
  "unit_id": "tatmadaw_lid_99",
  "official_name": "Light Infantry Division 99",
  "name_burmese": "ပေါ့ပါးကာကွယ်ရေးတပ်မတော် ၉၉",
  "common_names": ["LID 99", "99th Division", "Division 99"],
  "unit_type": "light_infantry_division",
  "parent_command": "Northern Command",
  "known_locations": ["Kale", "Sagaing"],
  "active_period": "1990s-present",
  "notes": "Frequent deployments in Sagaing Region post-coup"
}
```

**Matching Process**:
1. Extract candidate unit name from text
2. Normalize (remove punctuation, handle transliteration variants)
3. Search database by name/aliases
4. Score match similarity
5. If match > 0.85 → link to database entity
6. If no match → flag as "potential new unit" for verification

### Township/Village Database
- Hierarchical: Village → Township → Region
- Multiple spellings per location (Burmese transliteration varies)
- Geocoding (lat/lon when available)
- Merge/rename history (villages change names post-conflict)

### Person/Organization Database
- Known activists, military officers, officials
- Helps disambiguate common names
- Track aliases, positions, affiliations

---

## Temporal Extraction

### Myanmar Calendar Handling
```
Burmese: ၁၃၈၂ ခုနှစ် ကဆုန်လ ၁၅ ရက်
Interpretation: Year 1382 (Myanmar calendar), Month Kason, Day 15
Conversion: 2021-04-15 (Gregorian)
```

**Extraction Pipeline**:
1. Detect Myanmar calendar markers (month names, year format)
2. Extract year, month, day
3. Convert to Gregorian using `myanmar-calendar` library
4. Store both versions (original + Gregorian)
5. Note conversion confidence

### Fuzzy Temporal Markers
- "Last week" → Requires evidence date context
- "During harvest season" → Map to typical harvest months (Nov-Jan in Myanmar)
- "Before the coup" → Before 2021-02-01
- "A few days after the raid on Village X" → Requires cross-event linking

**Handling**:
- Create date ranges instead of exact dates
- Document interpretation logic
- Flag for human review if ambiguous

---

## Confidence Scoring

### Factors Increasing Confidence
- Multiple sources mention same entity (+0.1 per corroborating source, max +0.3)
- Matches context database exactly (+0.1)
- High OCR quality in source (+0.05)
- Rule-based AND ML methods agree (+0.1)
- Human verification (+0.2)

### Factors Decreasing Confidence
- Single source only (-0.1)
- Low OCR quality (-0.1)
- No context database match (-0.15)
- Methods disagree (-0.2)
- Ambiguous text (multiple interpretations) (-0.15)

### Base Confidences
- Rule-based match: 0.7
- ML extraction (high score): 0.8
- Context database match: 0.9
- Human annotation: 1.0

---

## Human-in-the-Loop Review

### Review Queue Items
- Confidence < threshold
- Contradictory extractions (ML says X, rules say Y)
- No context match found (potential new entity)
- Ambiguous text spans

### Review Interface
```bash
rise-extract --review-queue extracted_entities/review_queue/

# Shows:
# - Original text (Burmese + English translation)
# - Extracted entity candidate
# - Confidence scores
# - Reasoning
# - Context database suggestions
#
# Human actions:
# - Approve (boost confidence to 1.0)
# - Correct (modify extraction, set confidence 1.0)
# - Reject (remove extraction)
# - Flag for further investigation
```

---

## Integration with Downstream Tools

### Feed into Verification
```bash
rise-extract --input bundles/ --output extracted/
rise-verify --entities extracted/entities/ --events extracted/events/ --output verified/
```

### Re-extraction with Better Context
```bash
# Initial extraction
rise-extract --input bundles/ --context myanmar --output extracted_v1/

# Updated context library (more military units identified)
rise-extract --input bundles/ --context myanmar_v2 --output extracted_v2/

# Compare versions
diff extracted_v1/entities/ extracted_v2/entities/
```

---

## The Brilliant Prompt

> **You are a forensic intelligence analyst extracting structured data from Myanmar conflict evidence.**
>
> Mission: Identify entities (military units, persons, locations, organizations) and events (attacks, arrests, movements) from Viber screenshots, witness statements, and field reports.
>
> Critical rules:
> 1. **Never invent information**. Low confidence with documented uncertainty beats confident hallucination.
> 2. **Preserve Burmese terms**. Extract "လိုင်း ၉၉" AND translate to "LID 99" - both matter.
> 3. **Score every extraction** (0.0-1.0). Why are you confident? Document reasoning.
> 4. **Check context databases**. Does "LID 99" exist in military unit database? Cross-reference.
> 5. **Handle fuzzy data**. Date says "last week"? Create date range. Location unclear? Note ambiguity.
> 6. **Build relationships**. LID 99 attacked Village X → link perpetrator to event to location.
> 7. **Flag for review**. Uncertain? Create review queue item for human expert.
>
> You are extracting intelligence that will be used for legal accountability. Every extraction must be:
> - Traceable to source text (character offsets)
> - Confidence-scored with reasoning
> - Validated against known facts (context databases)
> - Documented with method (how did you extract this?)
>
> Output: Structured JSON entities/events with complete provenance. Downstream tools (verification, timeline construction, dossier compilation) depend on your extractions being truthful and complete.

---

*rise-extract transforms normalized evidence into structured intelligence. Quality here determines the integrity of all downstream analysis.*
