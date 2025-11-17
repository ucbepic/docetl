# rise-verify: Multi-Source Verification

## Mission

Cross-validate extracted entities and events against multiple sources, detect contradictions, adjust confidence scores based on corroboration, and create verification matrices showing what confirms/contradicts what.

## Core Philosophy

You are a **forensic fact-checker**. Single-source claims are inherently uncertain. Your job:
- Cross-reference every extraction against all available sources
- Boost confidence for corroborated facts
- Lower confidence for contradicted claims  - **Never hide conflicts** - document every contradiction
- Create verification matrices (source A says X, source B says Y, source C says Z)
- Flag single-source dependencies as verification priorities
- Build corroboration graphs

**Golden Rule: 100% conflict detection rate - never hide contradictions for convenience.**

---

## Usage

### Basic Cross-Source Verification
```bash
rise-verify \
  --entities extracted_entities/entities/ \
  --events extracted_events/events/ \
  --sources evidence_bundles/bundles/ \
  --output verified/ \
  --conflicts conflicts_report.json
```

### Verification with Known Facts Database
```bash
rise-verify \
  --entities extracted/ \
  --ground-truth known_facts_db.json \
  --boost-matched-facts \
  --output verified/
```

### Temporal Consistency Checking
```bash
rise-verify \
  --events extracted_events/ \
  --temporal-consistency \
  --flag-impossible-sequences \
  --output verified/
```

---

## Parameters

### Input Parameters

**`--entities <path>`** (required)
- Directory of extracted entities (from rise-extract)

**`--events <path>`** (required)
- Directory of extracted events

**`--sources <path>`** (required)
- Evidence bundles (original sources for cross-checking)

**`--ground-truth <path>`** (optional)
- Pre-verified facts database (high-confidence reference data)

### Verification Methods

**`--cross-source`** (default: enabled)
- Compare entity/event mentions across different sources
- Count corroborations and contradictions

**`--temporal-consistency`** (optional)
- Check if event sequence is logically possible
- Flag: "Event A at 14:00, Event B (50km away) at 14:15" - impossible travel time

**`--spatial-consistency`** (optional)
- Check if locations are consistent
- Flag: "Unit X in City A and City B simultaneously"

**`--entity-resolution`** (optional)
- Merge duplicate entities ("LID 99" vs "99th Division" → same entity)
- Confidence-based merging

### Confidence Adjustment

**`--boost-corroborated <float>`** (optional, default: +0.1)
- Increase confidence for multi-source facts

**`--lower-contradicted <float>`** (optional, default: -0.2)
- Decrease confidence for conflicting claims

**`--single-source-penalty <float>`** (optional, default: 0.0)
- Apply penalty to single-source extractions
- Or just flag them without adjusting score

### Output Parameters

**`--output <path>`** (required)
- Directory for verified entities/events

**`--conflicts <path>`** (optional)
- JSON report of all contradictions detected

**`--verification-matrix <path>`** (optional)
- CSV/JSON matrix: sources × entities, showing confirm/contradict/no-mention

**`--single-source-flags <path>`** (optional)
- List of entities/events with only one source (priority for additional verification)

---

## Output Structure

### Verified Entity Format

```json
{
  "entity_id": "ent_mil_001_verified",
  "original_extraction_id": "ent_mil_001",
  "entity_type": "military_unit",
  "verification_timestamp": "2024-01-11T09:00:00Z",
  "verification_tool": "rise-verify v1.2.0",

  "entity_data": {
    // Same as extracted entity
    "name_burmese": "လိုင်း ၉၉",
    "name_english": "LID 99"
  },

  "verification_status": {
    "status": "corroborated",  // corroborated | contradicted | single_source | uncertain
    "confidence_original": 0.95,
    "confidence_verified": 0.98,  // Boosted due to corroboration
    "confidence_change": +0.03,
    "confidence_factors": [
      {"factor": "multi_source_corroboration", "adjustment": +0.10, "details": "3 sources confirm"},
      {"factor": "ground_truth_match", "adjustment": +0.05, "details": "Matches known military unit database"},
      {"factor": "temporal_consistency", "adjustment": -0.02, "details": "Minor date discrepancy (2 days)"}
    ]
  },

  "source_analysis": {
    "total_sources": 3,
    "corroborating_sources": [
      {
        "evidence_id": "evd_042",
        "mention": "လိုင်း ၉၉ အမှတ် ၁၄:၀၀ နာရီတွင် ရွာကို ပစ်ခတ်ခဲ့သည်",
        "confidence": 0.95,
        "date": "2021-04-15"
      },
      {
        "evidence_id": "evd_118",
        "mention": "99th Division artillery attack",
        "confidence": 0.92,
        "date": "2021-04-15"
      },
      {
        "evidence_id": "evd_119",
        "mention": "LID 99 shelling confirmed by witness",
        "confidence": 0.88,
        "date": "2021-04-17"  // 2 days later - follow-up report
      }
    ],
    "contradicting_sources": [],
    "non_mentioning_sources": 215  // Other sources don't mention this unit (not a contradiction)
  },

  "conflict_analysis": {
    "conflicts_detected": 1,
    "conflicts": [
      {
        "conflict_type": "date_discrepancy",
        "severity": "minor",
        "description": "Source evd_118 says '2021-04-15', Source evd_119 says '2021-04-17'",
        "resolution": "evd_119 is follow-up report, not primary witness - kept 2021-04-15 as attack date",
        "confidence_impact": -0.02
      }
    ]
  },

  "single_source_flags": {
    "is_single_source": false,
    "verification_priority": "low"  // Already corroborated
  }
}
```

### Conflict Report Format

```json
{
  "conflict_report_id": "conflicts_20240111",
  "report_timestamp": "2024-01-11T09:30:00Z",
  "total_conflicts": 12,

  "conflicts": [
    {
      "conflict_id": "conf_001",
      "conflict_type": "entity_attribute_mismatch",
      "severity": "major",

      "entity_id": "ent_per_001",
      "attribute": "status",

      "conflicting_claims": [
        {
          "source": "evd_042",
          "claim": "injured",
          "confidence": 0.9
        },
        {
          "source": "evd_150",
          "claim": "killed",
          "confidence": 0.85
        }
      ],

      "analysis": {
        "likely_resolution": "injured_initially_later_died",
        "reasoning": "evd_042 from day of incident, evd_150 from 3 days later",
        "recommended_action": "Search for follow-up reports on victim status",
        "human_review_required": true
      },

      "impact": {
        "entities_affected": ["ent_per_001"],
        "events_affected": ["evt_attack_001"],
        "confidence_adjustments": [
          {"entity_id": "ent_per_001", "attribute": "status", "new_confidence": 0.7}
        ]
      }
    },

    {
      "conflict_id": "conf_002",
      "conflict_type": "event_date_mismatch",
      "severity": "critical",

      "event_id": "evt_attack_002",
      "attribute": "date",

      "conflicting_claims": [
        {"source": "evd_078", "claim": "2021-03-10", "confidence": 0.88},
        {"source": "evd_079", "claim": "2021-03-15", "confidence": 0.85},
        {"source": "evd_080", "claim": "2021-03-12", "confidence": 0.82}
      ],

      "analysis": {
        "likely_resolution": "multiple_related_events",
        "reasoning": "All sources describe similar attacks but on different dates - may be sustained operation",
        "recommended_action": "Split into separate events, cross-reference location and perpetrator details",
        "human_review_required": true
      }
    }
  ],

  "conflict_statistics": {
    "by_type": {
      "date_mismatch": 5,
      "location_mismatch": 2,
      "casualty_count_mismatch": 3,
      "entity_attribute_mismatch": 2
    },
    "by_severity": {
      "critical": 3,
      "major": 5,
      "minor": 4
    },
    "resolution_rate": {
      "auto_resolved": 4,
      "requires_human_review": 8
    }
  }
}
```

### Verification Matrix Format

```csv
evidence_id,entity_id,relationship,confidence,notes
evd_042,ent_mil_001,confirms,0.95,"Direct mention of LID 99"
evd_118,ent_mil_001,confirms,0.92,"English variant confirmation"
evd_119,ent_mil_001,confirms,0.88,"Witness corroboration"
evd_150,ent_mil_001,no_mention,0.0,"Source about different area"
evd_042,ent_per_001,confirms,0.90,"Mentions Ko Aung injured"
evd_150,ent_per_001,contradicts,0.85,"Claims Ko Aung killed - conflict!"
```

---

## Verification Methods

### Cross-Source Corroboration

**Process**:
1. For each extracted entity/event, find all sources that mention it
2. Compare attribute values (dates, locations, names, etc.)
3. Count agreements and disagreements
4. Adjust confidence based on corroboration strength

**Confidence Adjustment**:
```python
if corroborating_sources >= 3:
    confidence += 0.15  # Strong corroboration
elif corroborating_sources == 2:
    confidence += 0.10  # Moderate corroboration
elif corroborating_sources == 1:
    confidence -= 0.05  # Single source warning

if contradicting_sources > 0:
    confidence -= 0.20  # Conflicts lower confidence significantly
```

### Temporal Consistency Checking

**Checks**:
- Event sequence possible? (A before B, both before C)
- Time gaps realistic? (Travel time between locations)
- Date formats consistent? (No "February 30th")

**Example Inconsistency**:
```
Event A: "LID 99 at Kale Township at 14:00"
Event B: "LID 99 at Yinmabin Township at 14:30"
Distance: 75km
Travel time required: ~2 hours

Conclusion: Impossible - flag as temporal inconsistency
Possible explanations:
  - Different sub-units of LID 99
  - Time recording error
  - Date error (one event on different day)
```

### Spatial Consistency Checking

**Checks**:
- Entity can't be in two places simultaneously
- Location hierarchies valid (village must be in township)
- Coordinates match place names

**Example Inconsistency**:
```
Source A: "Attack on Thayetchaung Village, Kale Township"
Source B: "Attack on Thayetchaung Village, Yinmabin Township"

Check database: Which township contains Thayetchaung?
Result: Kale Township
Conclusion: Source B has location error
Action: Correct to Kale, note source B geographic uncertainty
```

### Entity Resolution (Deduplication)

**Challenge**: Same entity referenced differently
- "LID 99" vs "99th Division" vs "Light Infantry Division 99"
- "Ko Aung" vs "Aung" vs "U Aung"

**Resolution**:
1. Normalize names (remove punctuation, handle transliteration)
2. Check aliases in entity data
3. Compare contexts (same event, same location → likely same entity)
4. Confidence-based merging:
   - High confidence both extractions + context match → merge
   - Low confidence or contradictory attributes → keep separate, flag for review

---

## Ground Truth Integration

### Known Facts Database

Pre-verified high-confidence facts:
```json
{
  "ground_truth_db_id": "myanmar_tatmadaw_units_2024",
  "version": "2.1",
  "verified_date": "2024-01-01",

  "entities": [
    {
      "entity_id": "gt_mil_lid99",
      "entity_type": "military_unit",
      "name_english": "Light Infantry Division 99",
      "name_burmese": "ပေါ့ပါးကာကွယ်ရေးတပ်မတော် ၉၉",
      "verification_status": "confirmed_by_multiple_reputable_sources",
      "confidence": 1.0,
      "attributes": {
        "parent_command": "Northern Command",
        "known_locations": ["Kale", "Sagaing"],
        "active_since": "1990s"
      }
    }
  ]
}
```

**Usage**:
- Extract mentions "LID 99"
- Check against ground truth database
- Match found → boost confidence (+0.05 to +0.10)
- Attribute mismatch → investigate (extracted claim contradicts known fact)

---

## Conflict Resolution Strategies

### Automatic Resolution (Low Severity)

**Scenario: Minor date discrepancy (1-2 days)**
- Source A: "2021-04-15"
- Source B: "2021-04-17"

**Resolution**:
- Check if Source B is follow-up report (mentions "earlier this week", etc.)
- If yes: Keep primary source date, note secondary source as corroboration
- If no: Create date range "2021-04-15 to 2021-04-17", flag for review

### Human Review Required (High Severity)

**Scenario: Casualty count discrepancy**
- Source A: "3 killed"
- Source B: "15 killed"

**Action**:
- Cannot auto-resolve - numbers too different
- Create conflict report entry
- Flag both sources for human review
- Lower confidence on casualty count to 0.6
- Note: "Significant source disagreement on casualties"

### Contextual Resolution

**Scenario: "LID 99 attacked two villages"**
- Source A: "LID 99 attacked Village X on April 15"
- Source B: "LID 99 attacked Village Y on April 15"

**Resolution**:
- NOT a contradiction - LID 99 could attack multiple villages same day
- Create two separate events
- Cross-reference both to LID 99
- Check distance between villages (if too far, may indicate sub-units)

---

## Single-Source Dependency Flagging

### What Is Single-Source Dependency?

Entity/event mentioned by **only one source** - inherently higher uncertainty.

### Flagging Process
```json
{
  "entity_id": "ent_mil_023",
  "single_source_flag": {
    "is_single_source": true,
    "source_id": "evd_209",
    "confidence_penalty": -0.10,
    "verification_priority": "high",
    "collection_recommendations": [
      "Search for satellite imagery of claimed location/date",
      "Check Tatmadaw social media for unit movements",
      "Interview additional witnesses from area"
    ]
  }
}
```

### Priority Ranking
- **Critical**: Major events (mass casualties) from single source
- **High**: New military units, commander names, specific tactics
- **Medium**: Common events (village raids) from single source
- **Low**: Background context, widely known facts

---

## Integration with Downstream Tools

### Feed into Timeline
```bash
rise-verify --entities extracted/ --events extracted/ --output verified/
rise-timeline --verified-entities verified/entities/ --verified-events verified/events/ --output timeline.json
```

### Iterative Verification (New Sources Added)
```bash
# Initial verification
rise-verify --entities extracted/ --events extracted/ --sources evidence_bundles/ --output verified_v1/

# New sources arrive
rise-ingest --source new_batch/ --output evidence_bundles_v2/
rise-extract --input evidence_bundles_v2/ --output extracted_v2/

# Re-verify with expanded source base
rise-verify \
  --entities verified_v1/entities/ \
  --entities extracted_v2/entities/ \
  --events verified_v1/events/ \
  --events extracted_v2/events/ \
  --sources evidence_bundles/ \
  --sources evidence_bundles_v2/ \
  --output verified_v2/

# Compare: how many single-source flags resolved?
diff verified_v1/single_source_flags.json verified_v2/single_source_flags.json
```

---

## The Brilliant Prompt

> **You are a forensic fact-checker verifying Myanmar conflict intelligence.**
>
> Mission: Cross-validate every extracted entity and event against all available sources. Never hide conflicts - document every contradiction.
>
> Core responsibilities:
> 1. **Find corroborations** - Multiple sources say "LID 99 attacked on April 15" → boost confidence
> 2. **Detect contradictions** - Source A says "3 killed", Source B says "15 killed" → CREATE CONFLICT REPORT
> 3. **Check consistency** - Event A at 14:00 in Kale, Event B at 14:15 in Yinmabin (75km away) → impossible, flag temporal inconsistency
> 4. **Validate against known facts** - Extracted "LID 99" matches ground truth database → boost confidence
> 5. **Flag single-source claims** - Only one source mentions this → mark as verification priority
> 6. **Build verification matrices** - Which sources confirm/contradict what
>
> Never resolve high-severity conflicts automatically. Create review queue for human experts.
>
> Confidence adjustment rules:
> - 3+ sources corroborate → +0.15
> - Ground truth match → +0.05
> - Any contradiction → -0.20
> - Single source only → flag (optional -0.10 penalty)
>
> Your output: Verified entities/events with adjusted confidence + conflict reports + verification matrices. Legal investigators rely on your conflict detection being 100% accurate.

---

*rise-verify is the truth-testing phase. Everything downstream depends on conflicts being detected, not hidden.*
