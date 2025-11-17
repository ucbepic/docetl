# rise-archive & rise-gap: Preservation and Gap Analysis

This document covers two critical RISE tools: **rise-archive** (primary source preservation with chain of custody) and **rise-gap** (intelligence gap identification and prioritization).

---

# rise-archive: Primary Source Preservation

## Mission

Preserve original evidence with forensic-grade chain of custody, enable future re-analysis, enforce security protocols (embargo, anonymization), and maintain complete provenance from source to intelligence product.

## Core Philosophy

You are a **forensic archivist**. Your job is long-term preservation and integrity:
- Store originals bit-for-bit (hash-verified)
- Document every transformation (ingestion, extraction, verification)
- Enforce security (embargo dates, access controls, anonymization verification)
- Enable re-analysis (years later, with better tools)
- Meet legal evidence standards (admissible in court)

**Rule: Archives are immutable. Once archived, read-only. Never modify archived evidence.**

---

## Usage

### Basic Archiving
```bash
rise-archive \
  --evidence evidence_bundles/ \
  --output forensic_archive/ \
  --format immutable
```

### Secure Archive with Encryption
```bash
rise-archive \
  --evidence evidence_bundles/ \
  --verified verified/ \
  --output secure_archive/ \
  --encrypt \
  --embargo-enforce \
  --access-log archive_access.log
```

### Archive with Re-Analysis Package
```bash
rise-archive \
  --evidence bundles/ \
  --verified verified/ \
  --timeline timeline.json \
  --dossiers dossiers/ \
  --output complete_archive/ \
  --include-processing-history \
  --reproducibility-package
```

---

## Parameters

### Input Parameters

**`--evidence <path>`** (required)
- Evidence bundles (from rise-ingest)
- Original files + normalized versions + metadata

**`--verified <path>`** (optional)
- Verified entities/events (from rise-verify)
- Links evidence → intelligence products

**`--timeline <path>`** (optional)
- Timeline JSON for cross-referencing

**`--dossiers <path>`** (optional)
- Dossier outputs for cross-referencing

### Archive Configuration

**`--format <type>`** (optional, default: immutable)
- `immutable`: Write-once, read-many (WORM)
- `versioned`: Allow updates but track all versions
- `redundant`: Multiple copies for disaster recovery

**`--encrypt`** (optional)
- Encrypt archived evidence (AES-256)
- Requires key management configuration

**`--embargo-enforce`** (optional)
- Check embargo dates, prevent access to embargoed evidence
- Alert on attempted unauthorized access

**`--compression <method>`** (optional)
- `none`: No compression (preserve exact bytes)
- `lossless`: Gzip/LZMA (reduce storage, preserve data)
- Default: `lossless` for non-binary formats

### Security & Access

**`--access-log <path>`** (optional)
- Log every access attempt
- Format: timestamp, user, evidence_id, action, granted/denied

**`--access-control <policy>`** (optional)
- Define who can access what
- Roles: `researcher`, `legal_team`, `admin`

**`--anonymization-verify`** (optional)
- Verify anonymization was applied correctly
- Check: faces blurred, phone numbers redacted, etc.

### Output Parameters

**`--output <path>`** (required)
- Archive directory structure

**`--include-processing-history`** (optional)
- Store all intermediate processing outputs
- Enables re-analysis and methodology audit

**`--reproducibility-package`** (optional)
- Package everything needed to reproduce analysis:
  - Tool versions
  - Configuration files
  - Context databases (snapshot at time of processing)
  - Processing logs

---

## Archive Structure

```
forensic_archive/
├── originals/
│   ├── evd_20240110_142300_001_original.jpg
│   ├── evd_20240110_142300_002_original.pdf
│   └── ...
├── evidence_bundles/
│   ├── evd_20240110_142300_001.json
│   ├── evd_20240110_142300_002.json
│   └── ...
├── verified/
│   ├── entities/
│   │   └── ent_mil_001_verified.json
│   └── events/
│       └── evt_attack_001_verified.json
├── outputs/
│   ├── timeline.json
│   ├── dossiers/
│   │   └── military_operations.json
│   └── gap_analysis.json
├── provenance/
│   ├── provenance_ledger.jsonl  # Append-only log
│   ├── chain_of_custody/
│   │   └── evd_042_custody_chain.json
│   └── processing_history/
│       └── evd_042_processing.json
├── security/
│   ├── embargo_registry.json
│   ├── access_log.jsonl
│   ├── anonymization_log.json
│   └── encryption_keys/  # Secure key storage
├── reproducibility/
│   ├── tool_versions.json
│   ├── configuration/
│   │   ├── rise-ingest-config.json
│   │   └── rise-extract-config.json
│   ├── context_databases/
│   │   ├── myanmar_tatmadaw_2024_snapshot.db
│   │   └── townships_2024_snapshot.db
│   └── processing_logs/
│       └── full_pipeline_log.txt
└── README.md  # Archive documentation
```

---

## Provenance Ledger

**Append-only log of every action**:

```jsonl
{"timestamp":"2024-01-10T14:23:00Z","action":"ingest","evidence_id":"evd_042","actor":"rise-ingest v1.2.0","details":"Original file ingested, Zawgyi detected, converted to Unicode"}
{"timestamp":"2024-01-10T15:30:00Z","action":"extract","evidence_id":"evd_042","actor":"rise-extract v1.2.0","details":"Extracted entity ent_mil_001 (LID 99) with confidence 0.95"}
{"timestamp":"2024-01-11T09:00:00Z","action":"verify","entity_id":"ent_mil_001","actor":"rise-verify v1.2.0","details":"Corroborated by evd_118, evd_119, confidence increased to 0.98"}
{"timestamp":"2024-01-11T10:00:00Z","action":"timeline_add","event_id":"evt_attack_001","actor":"rise-timeline v1.2.0","details":"Added to timeline at sequence 234"}
{"timestamp":"2024-01-11T11:00:00Z","action":"dossier_add","event_id":"evt_attack_001","actor":"rise-dossier v1.2.0","details":"Included in military_operations dossier"}
{"timestamp":"2024-01-11T14:00:00Z","action":"archive","evidence_id":"evd_042","actor":"rise-archive v1.2.0","details":"Archived with hash verification, embargo until 2026-01-10"}
{"timestamp":"2024-02-15T16:30:00Z","action":"access","evidence_id":"evd_042","actor":"researcher_jane@org.org","details":"Accessed for legal case preparation","granted":true}
{"timestamp":"2024-03-01T10:00:00Z","action":"access","evidence_id":"evd_042","actor":"unauthorized_user@external.com","details":"Attempted access to embargoed evidence","granted":false}
```

Every action, every actor, every timestamp. Complete audit trail.

---

## Reproducibility Package

**Enable re-analysis years later**:

```json
{
  "archive_id": "myanmar_sagaing_2021_2024_archive",
  "creation_date": "2024-01-11",
  "tool_versions": {
    "rise-ingest": "1.2.0",
    "rise-extract": "1.2.0",
    "rise-verify": "1.2.0",
    "rise-timeline": "1.2.0",
    "rise-dossier": "1.2.0",
    "rise-archive": "1.2.0"
  },
  "context_databases": {
    "myanmar_tatmadaw_2024": {
      "version": "2.1",
      "snapshot_date": "2024-01-01",
      "path": "reproducibility/context_databases/myanmar_tatmadaw_2024_snapshot.db",
      "hash": "sha256:abcdef123456..."
    },
    "myanmar_townships": {
      "version": "1.5",
      "snapshot_date": "2024-01-01",
      "path": "reproducibility/context_databases/townships_2024_snapshot.db",
      "hash": "sha256:123456abcdef..."
    }
  },
  "configuration_files": [
    "rise-ingest-config.json",
    "rise-extract-config.json",
    "brilliant-prompts/ingestion-briefing.md"
  ],
  "re_analysis_instructions": "To reproduce this analysis: 1) Use tool versions specified above, 2) Load context database snapshots, 3) Apply same configuration files, 4) Process evidence bundles in order, 5) Compare outputs to archived outputs"
}
```

Future researchers can re-run analysis with identical tools/context.

---

# rise-gap: Intelligence Gap Analysis

## Mission

Identify what we don't know - temporal gaps (missing documentation periods), geographic gaps (uncovered areas), single-source dependencies, verification needs - and prioritize intelligence collection efforts.

## Core Philosophy

You are a **forensic gap analyst**. Intelligence value = knowing what we don't know:
- Temporal gaps: No events documented for 2 weeks in high-conflict area? Missing data or quiet period?
- Geographic gaps: Township A heavily documented, Township B (adjacent) has nothing? Suspicious.
- Single-source claims: Major event from only one source? Needs corroboration.
- Entity gaps: "LID 99 attacked" but commander name unknown? Limits accountability.

**Rule: Explicit gaps > assumed completeness. Never pretend we know more than we do.**

---

## Usage

### Basic Gap Analysis
```bash
rise-gap \
  --timeline timeline.json \
  --verified verified/ \
  --output gap_analysis.json
```

### Comprehensive Gap Report
```bash
rise-gap \
  --timeline timeline.json \
  --verified verified/ \
  --dossiers dossiers/ \
  --context myanmar_sagaing \
  --output gap_report.pdf \
  --prioritize
```

### Geographic Gap Focus
```bash
rise-gap \
  --timeline timeline.json \
  --geographic-analysis \
  --compare-adjacent-townships \
  --output geographic_gaps.json
```

---

## Parameters

### Input Parameters

**`--timeline <path>`** (required)
- Timeline JSON for temporal gap analysis

**`--verified <path>`** (required)
- Verified entities/events for verification gap analysis

**`--dossiers <path>`** (optional)
- Dossiers for thematic gap analysis

**`--context <name>`** (required)
- Context library for expected coverage
- Example: myanmar_sagaing defines townships, conflict intensity

### Analysis Types

**`--temporal-gaps`** (default: enabled)
- Identify periods with missing documentation

**`--geographic-gaps`** (optional)
- Compare coverage across geographic areas

**`--verification-gaps`** (optional)
- Identify single-source claims, unverified events

**`--entity-gaps`** (optional)
- Identify incomplete entity information
- Example: Military unit mentioned but commander unknown

**`--thematic-gaps`** (optional)
- Analyze gaps in thematic coverage
- Example: Many military ops documented, little on displacement

### Prioritization

**`--prioritize`** (optional)
- Rank gaps by collection priority
- Factors: Severity, location importance, evidence type

**`--collection-recommendations`** (optional)
- Suggest specific actions to fill gaps
- "Interview IDPs from Township X during Period Y"

### Output Parameters

**`--output <path>`** (required)
- Gap analysis output
- Format: `.json`, `.pdf`, `.html`

**`--format <type>`** (optional, default: json)
- `json`: Structured gap data
- `pdf`: Gap report for planning
- `html`: Interactive gap visualization

---

## Gap Types

### 1. Temporal Gaps

```json
{
  "gap_type": "temporal",
  "gap_id": "tgap_001",
  "start_date": "2021-03-15",
  "end_date": "2021-03-22",
  "duration_days": 7,
  "location": "Kale_Township",

  "context": {
    "events_before": "Heavy military operations",
    "events_after": "Mass displacement reported",
    "known_indicators": "Satellite imagery shows village burning during gap period",
    "conclusion": "Missing documentation, not quiet period"
  },

  "priority": "critical",
  "priority_reasoning": "Known operations occurred but no ground documentation - evidence gap for legal case",

  "collection_recommendations": [
    "Interview IDPs who fled Kale Township during March 15-22",
    "Review satellite imagery archives for this period",
    "Contact local CDM networks for any documentation from this week",
    "Check social media archives (Facebook, Viber groups) for March 15-22 posts"
  ]
}
```

### 2. Geographic Gaps

```json
{
  "gap_type": "geographic",
  "gap_id": "ggap_001",
  "location": "Yinmabin_Township_Southern_Villages",

  "coverage_analysis": {
    "this_area_events": 23,
    "this_area_event_density": 0.3,  // events per village
    "adjacent_area_events": 234,
    "adjacent_area_event_density": 4.2,
    "expected_density": 3.5  // Based on conflict intensity indicators
  },

  "gap_assessment": "Significant undercoverage - expected ~245 events based on conflict indicators, only 23 documented",

  "possible_reasons": [
    "Limited documentation network access",
    "Military control prevents documentation",
    "Documentation exists but not yet integrated",
    "Genuinely lower conflict intensity (less likely given indicators)"
  ],

  "priority": "high",
  "collection_recommendations": [
    "Establish contact with documentation networks in southern Yinmabin",
    "Interview IDPs from this area now in camps",
    "Review remote sensing data for signs of conflict activity",
    "Cross-reference with neighboring township documentation for spillover events"
  ]
}
```

### 3. Verification Gaps (Single-Source Claims)

```json
{
  "gap_type": "verification",
  "gap_id": "vgap_001",
  "claim_type": "major_incident",

  "incident": {
    "event_id": "evt_attack_445",
    "description": "Artillery attack on Kawlin Township, 15 casualties claimed",
    "date": "2022-08-10",
    "source": "evd_209",
    "source_reliability": "unknown"
  },

  "verification_status": {
    "sources_count": 1,
    "corroboration": "none",
    "contradictions": "none",
    "confidence": 0.65
  },

  "verification_attempts": [
    "Searched satellite imagery - inconclusive due to cloud cover",
    "Checked Tatmadaw social media - no mention (expected)",
    "Searched for witness statements - none found",
    "Contacted local networks - no response yet"
  ],

  "priority": "critical",
  "priority_reasoning": "15 casualties is major incident, but single-source claim reduces confidence - needs corroboration for legal use",

  "collection_recommendations": [
    "Field investigation if area accessible",
    "Interview medical staff from Kawlin for casualty records",
    "Search for additional witnesses via social networks",
    "Request satellite imagery review for Aug 9-11 period"
  ]
}
```

### 4. Entity Gaps

```json
{
  "gap_type": "entity",
  "gap_id": "egap_001",
  "entity_id": "ent_mil_001",
  "entity_type": "military_unit",

  "known_information": {
    "name": "Light Infantry Division 99",
    "activities": "234 documented attacks",
    "locations": ["Kale", "Yinmabin", "Tamu"],
    "active_period": "2021-02-01 to present"
  },

  "missing_information": [
    {
      "attribute": "commander_name",
      "importance": "critical",
      "reasoning": "Required for command responsibility determination",
      "period": "2021-2022"
    },
    {
      "attribute": "battalion_structure",
      "importance": "high",
      "reasoning": "Helps attribute specific attacks to sub-units"
    },
    {
      "attribute": "base_location",
      "importance": "medium",
      "reasoning": "Context for operational patterns"
    }
  ],

  "priority": "high",
  "collection_recommendations": [
    "Review Tatmadaw organizational documents for LID 99 command structure",
    "Interview defected soldiers from LID 99",
    "Analyze social media for commanders' public appearances",
    "Cross-reference with military promotion announcements"
  ]
}
```

---

## Prioritization Algorithm

```python
def prioritize_gap(gap):
    score = 0

    # Severity
    if gap.type == "verification" and gap.incident.casualties > 10:
        score += 5  # Major incident needs verification
    if gap.type == "temporal" and gap.duration_days > 14:
        score += 3  # Long gap is suspicious

    # Evidence type
    if gap.type == "verification" and gap.sources_count == 1:
        score += 3  # Single-source is risky
    if gap.type == "entity" and gap.missing_info == "commander_name":
        score += 4  # Command responsibility crucial

    # Location importance
    if gap.location in high_conflict_townships:
        score += 2
    if gap.location == "Kale_Township":  # Known hotspot
        score += 1

    # Collection feasibility
    if gap.location_accessible:
        score += 1  # Can actually collect
    else:
        score -= 2  # Hard to fill, lower priority

    # Priority assignment
    if score >= 8:
        return "critical"
    elif score >= 5:
        return "high"
    elif score >= 3:
        return "medium"
    else:
        return "low"
```

---

## Collection Planning Output

```json
{
  "collection_plan": {
    "plan_id": "collection_plan_2024_q2",
    "creation_date": "2024-01-11",

    "priority_gaps": [
      {
        "gap_id": "vgap_001",
        "gap_type": "verification",
        "priority": "critical",
        "action": "Field investigation - Kawlin Township artillery attack verification",
        "resources_needed": ["Field team", "Medical records access", "Satellite imagery"],
        "timeline": "Within 30 days",
        "assigned_to": "Field Team Alpha"
      },
      {
        "gap_id": "tgap_001",
        "gap_type": "temporal",
        "priority": "critical",
        "action": "IDP interviews - Kale Township March 15-22 period",
        "resources_needed": ["Interview team", "IDP camp access"],
        "timeline": "Within 60 days",
        "assigned_to": "Interview Team Beta"
      }
    ],

    "geographic_focus": [
      "Yinmabin_Township_Southern_Villages - Establish documentation network",
      "Kawlin_Township - Follow up on single-source claims"
    ],

    "thematic_focus": [
      "Displacement patterns - Underdocumented relative to military operations",
      "Economic impact - Very few entries, needs systematic collection"
    ]
  }
}
```

---

## Integration with Other Tools

```bash
# Full pipeline with gap analysis
rise-compose \
  --ingest sources/ \
  --extract all \
  --verify cross-source \
  --timeline monthly \
  --dossier themes=all \
  --gap-analysis \
  --output analysis_2024/

# Gap analysis guides next collection round
cat analysis_2024/gap_analysis.json | jq '.priority_gaps[] | select(.priority=="critical")'

# Ingest new sources targeting gaps
rise-ingest --source new_collection_kale_march/ --output evidence_bundles_v2/

# Re-run analysis - how many gaps filled?
rise-gap --timeline timeline_v2.json --verified verified_v2/ --compare-to analysis_2024/gap_analysis.json
```

---

## The Brilliant Prompts

### rise-archive Prompt

> **You are a forensic archivist preserving conflict documentation for long-term use.**
>
> Mission: Archive original evidence with complete chain of custody, enforce security protocols, enable future re-analysis.
>
> Critical tasks:
> 1. **Immutability** - Once archived, read-only. Never modify archived evidence.
> 2. **Hash verification** - Verify original files match ingestion hashes. Detect any corruption.
> 3. **Provenance ledger** - Append-only log of every action. Who accessed what when, what processing occurred.
> 4. **Embargo enforcement** - Check embargo dates, prevent unauthorized access, log violations.
> 5. **Reproducibility** - Package tool versions, configurations, context databases so analysis can be re-run years later.
>
> Success: 10 years from now, researcher can verify original evidence integrity, understand complete processing history, re-run analysis with same tools/context.

### rise-gap Prompt

> **You are a forensic gap analyst identifying what we don't know.**
>
> Mission: Find temporal gaps, geographic gaps, single-source claims, missing entity details - and prioritize collection efforts.
>
> Critical tasks:
> 1. **Temporal gaps** - No events for 2 weeks in conflict zone? Missing data or quiet period? Check satellite imagery, adjacent areas for clues.
> 2. **Geographic gaps** - Township A: 200 events. Township B (adjacent, similar): 10 events. SUSPICIOUS - likely documentation gap.
> 3. **Single-source risks** - Major incident from one source? Flag for corroboration. Legal cases need multi-source verification.
> 4. **Entity gaps** - "LID 99 attacked" but commander unknown? Limits accountability - high priority to identify.
> 5. **Prioritize** - Critical gaps (major incidents, legal needs) get top priority. Low-priority gaps noted but not urgent.
> 6. **Collection guidance** - Don't just identify gaps - suggest HOW to fill them. "Interview IDPs from Township X Period Y."
>
> Output: Gap analysis with priorities, collection plan, feasibility assessment.
>
> Success: Documentation team knows exactly where to focus next collection efforts, legal team knows which claims need more evidence.

---

*rise-archive ensures forensic integrity for the long term. rise-gap ensures we know what we don't know and can plan to find out.*
