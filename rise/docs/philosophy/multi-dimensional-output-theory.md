# Multi-Dimensional Output Theory

## The Core Problem

Traditional data pipelines produce **single linear outputs**:
- A database table
- A JSON file
- A CSV export
- A report document

But conflict documentation serves **radically different users** with **incompatible needs**:

| User | Primary Need | Ideal Format | Key Features |
|------|-------------|--------------|--------------|
| Legal investigators | Chronological fact timeline | Date-sorted log | Every event with source citations |
| Thematic researchers | Topic-organized intelligence | Dossiers by theme | Military ops separate from displacement |
| Verification teams | Primary source access | Original evidence archive | Chain of custody, bit-level preservation |
| Intelligence analysts | Gap identification | Missing information map | What we don't know, where to look next |
| Field teams | Geographic patterns | Location-based clusters | Township-level aggregation |
| External auditors | Methodology transparency | Provenance chains | Every decision documented |

**A single output format cannot serve these needs.** Attempts to create "one format to rule them all" result in:
- Compromises that serve no one well
- Information loss (hiding details some users need)
- Information overload (showing details others don't need)
- Unusable complexity (trying to be everything to everyone)

## The RISE Solution: Parallel Output Streams

RISE generates **four core output dimensions** simultaneously from the same verified evidence base:

```
                    VERIFIED EVIDENCE BASE
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   TEMPORAL            THEMATIC           PRESERVATION
   DIMENSION           DIMENSION           DIMENSION
        │                   │                   │
  Master Timeline    Thematic Dossiers   Primary Archives
  (chronological)    (topic-organized)    (originals + chain)
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                    ANALYTICAL DIMENSION
                            │
                      Gap Analysis
                   (what's missing)
```

Each dimension serves distinct investigative functions while remaining **synchronized** through shared entity IDs and provenance chains.

---

## Dimension 1: TEMPORAL (Master Timeline)

**Purpose**: Answer "What happened when?"

### Structure
```json
{
  "timeline_id": "myanmar_sagaing_2021_2024",
  "events": [
    {
      "event_id": "evt_001",
      "date": "2021-02-01",
      "date_precision": "exact|approximate|range",
      "description": "Military coup announced via MRTV",
      "entities_involved": [
        {"type": "organization", "id": "org_tatmadaw", "role": "perpetrator"},
        {"type": "person", "id": "per_min_aung_hlaing", "role": "leader"}
      ],
      "locations": [
        {"type": "city", "id": "loc_yangon"}
      ],
      "sources": [
        {"evidence_id": "evd_042", "confidence": 1.0}
      ],
      "thematic_tags": ["coup", "military_takeover"],
      "next_event": "evt_002",
      "prev_event": null
    }
  ],
  "temporal_gaps": [
    {
      "start_date": "2021-03-15",
      "end_date": "2021-03-22",
      "location": "Sagaing_Kale_Township",
      "reason": "No sources available for this period",
      "priority": "high"
    }
  ]
}
```

### Key Features
- **Strict chronological ordering**
- **Explicit date precision** (we know the difference between "February 2021" and "February 1, 2021, 06:30")
- **Linked events** (previous/next for narrative flow)
- **Temporal gap identification** (missing periods flagged)
- **Source attribution per event**

### Use Cases
- Legal chronologies for court submissions
- Timeline visualizations
- Temporal pattern analysis (escalation, cycles)
- "What was happening when" queries

---

## Dimension 2: THEMATIC (Topic Dossiers)

**Purpose**: Answer "Tell me everything about X"

### Structure
Multiple dossiers, each focused on a specific theme:

```json
{
  "dossier_id": "theme_military_operations_sagaing",
  "theme": "Military Operations",
  "subthemes": ["artillery_attacks", "village_raids", "aerial_bombardment"],
  "geographic_scope": "Sagaing_Region",
  "temporal_scope": "2021-02-01_to_2024-12-31",

  "entries": [
    {
      "entry_id": "dsr_mil_001",
      "incident_type": "artillery_attack",
      "date": "2021-04-15",
      "location": "Kale_Township_Thayetchaung_Village",
      "summary": "LID 99 Artillery Unit shelled village at 14:00",
      "entities": {
        "perpetrators": [
          {"type": "military_unit", "id": "unit_lid99_arty", "confidence": 0.95}
        ],
        "victims": [
          {"type": "person", "id": "per_ko_aung", "status": "injured", "confidence": 0.9}
        ]
      },
      "evidence": [
        {"id": "evd_117", "type": "photo", "description": "Shell crater"},
        {"id": "evd_118", "type": "witness_statement"}
      ],
      "cross_references": {
        "timeline_event": "evt_234",
        "related_dossiers": ["theme_displacement", "theme_violations"],
        "geographic_index": "geo_kale_township"
      }
    }
  ],

  "summary_statistics": {
    "total_incidents": 147,
    "date_range": "2021-02-01 to 2024-10-15",
    "locations_affected": 42,
    "civilian_casualties": {"killed": 23, "injured": 67},
    "military_units_identified": ["LID 99", "LID 88", "LID 77"]
  }
}
```

### Standard Dossier Types
1. **Military Operations** - Organized armed actions
2. **Human Rights Violations** - Attacks on civilians, detentions, torture
3. **Displacement Patterns** - Forced migration, IDP camps, refugee flows
4. **Economic Impact** - Looting, destruction, livelihood disruption
5. **Resistance Activities** - PDF operations, civil disobedience
6. **Political Developments** - CRPH, NUG, parallel government
7. **Humanitarian Response** - Aid delivery, health services, protection
8. **Media & Information** - Propaganda, censorship, documentation efforts

### Key Features
- **Topic-focused organization** (all military ops together)
- **Cross-referenced to timeline** (can jump to chronological context)
- **Summary statistics** (quantitative patterns)
- **Multi-dossier entries** (same incident appears in multiple relevant dossiers)

### Use Cases
- Thematic research papers
- Human rights reports organized by violation type
- Military pattern analysis
- Sector-specific advocacy (displacement, health, etc.)

---

## Dimension 3: PRESERVATION (Primary Source Archives)

**Purpose**: Answer "Show me the original evidence"

### Structure
```json
{
  "archive_id": "archive_myanmar_sagaing_2024",
  "evidence_items": [
    {
      "evidence_id": "evd_042",
      "original_file": {
        "filename": "viber_screenshot_20210415.jpg",
        "format": "JPEG",
        "size_bytes": 1247893,
        "hash_sha256": "a3f7c8e9...",
        "ingestion_date": "2024-01-10T14:23:00Z"
      },

      "chain_of_custody": [
        {
          "timestamp": "2021-04-15T16:30:00+06:30",
          "action": "created",
          "actor": "original_source",
          "details": "Screenshot taken in Viber app"
        },
        {
          "timestamp": "2024-01-10T08:00:00Z",
          "action": "collected",
          "actor": "field_researcher_042",
          "method": "secure_transfer"
        },
        {
          "timestamp": "2024-01-10T14:23:00Z",
          "action": "ingested",
          "actor": "rise_ingest_v1.2.0",
          "details": "Encoding: Zawgyi detected and converted to Unicode"
        }
      ],

      "processing_history": [
        {
          "timestamp": "2024-01-10T14:25:00Z",
          "tool": "rise_extract_v1.2.0",
          "action": "entity_extraction",
          "entities_extracted": ["unit_lid99_arty", "loc_kale_township"],
          "confidence_scores": [0.95, 0.98]
        },
        {
          "timestamp": "2024-01-11T09:00:00Z",
          "tool": "rise_verify_v1.2.0",
          "action": "cross_source_verification",
          "sources_compared": ["evd_042", "evd_118", "evd_119"],
          "result": "corroborated"
        }
      ],

      "security_metadata": {
        "embargo_until": "2026-01-10",
        "anonymization_applied": true,
        "sensitive_elements": ["faces_blurred", "phone_numbers_redacted"],
        "access_restrictions": "verified_researchers_only"
      },

      "extracted_content": {
        "text_original_zawgyi": "[original Zawgyi bytes]",
        "text_normalized_unicode": "လိုင်း ၉၉ အမှတ် ၁၄:၀၀ နာရီတွင် ရွာကို ပစ်ခတ်ခဲ့သည်",
        "ocr_confidence": 0.92,
        "translation_en": "LID 99 shelled the village at 14:00"
      },

      "referenced_by": [
        {"type": "timeline_event", "id": "evt_234"},
        {"type": "dossier_entry", "id": "dsr_mil_001"},
        {"type": "verification_matrix", "id": "vrfy_042_118_119"}
      ]
    }
  ]
}
```

### Key Features
- **Bit-level preservation** of originals (hash-verified)
- **Complete chain of custody** (every action logged)
- **Processing transparency** (AI models, human edits, versioning)
- **Security enforcement** (embargo, anonymization, access control)
- **Forward traceability** (what analysis used this evidence)

### Use Cases
- Forensic evidence presentation
- Methodology audits
- Re-analysis with improved tools
- Legal evidence chains
- Source protection verification

---

## Dimension 4: ANALYTICAL (Gap Analysis)

**Purpose**: Answer "What don't we know? Where should we look?"

### Structure
```json
{
  "gap_analysis_id": "gaps_sagaing_2024_q1",
  "analysis_date": "2024-03-31",
  "evidence_base_summary": {
    "total_events": 1247,
    "date_range": "2021-02-01 to 2024-03-31",
    "geographic_coverage": "Sagaing_Region",
    "source_count": 342
  },

  "temporal_gaps": [
    {
      "gap_id": "tgap_001",
      "type": "missing_period",
      "start_date": "2021-03-15",
      "end_date": "2021-03-22",
      "location": "Kale_Township",
      "context": "Known military operation in area, but no documentation",
      "priority": "high",
      "collection_suggestions": [
        "Contact Kale-based CDM networks",
        "Review social media from this period",
        "Interview IDPs who fled during this time"
      ]
    }
  ],

  "geographic_gaps": [
    {
      "gap_id": "ggap_001",
      "type": "underrepresented_area",
      "location": "Yinmabin_Township_Southern_Villages",
      "event_density": 0.3,  // events per village
      "comparison_density": 4.2,  // average for similar townships
      "reason": "Limited source access due to military control",
      "priority": "medium"
    }
  ],

  "thematic_gaps": [
    {
      "gap_id": "thgap_001",
      "type": "single_source_dependency",
      "theme": "LID_77_operations",
      "incidents": 23,
      "unique_sources": 1,
      "risk": "All information from single source - needs corroboration",
      "priority": "high"
    }
  ],

  "verification_gaps": [
    {
      "gap_id": "vgap_001",
      "type": "unverified_claims",
      "incident_id": "evt_445",
      "claim": "Artillery attack on Kawlin Township, 15 casualties",
      "source": "evd_209",
      "attempts": [
        "Searched satellite imagery - inconclusive cloud cover",
        "Checked Tatmadaw social media - no mention",
        "No corroborating witness statements found"
      ],
      "status": "requires_field_investigation"
    }
  ],

  "entity_gaps": [
    {
      "gap_id": "egap_001",
      "type": "incomplete_military_unit",
      "entity_id": "unit_lid88_battalion_unknown",
      "known_info": {
        "parent_unit": "LID_88",
        "location": "Kale_Township",
        "activity": "village_raids"
      },
      "missing_info": [
        "Battalion number",
        "Commanding officer",
        "Base location",
        "Subordinate companies"
      ],
      "collection_priority": "medium"
    }
  ]
}
```

### Key Features
- **Explicit gap identification** (what's missing)
- **Prioritization** (high/medium/low based on investigative value)
- **Collection suggestions** (where to look next)
- **Density analysis** (compare coverage across areas/themes)
- **Single-source risks** (dependencies on unverified claims)

### Use Cases
- Intelligence collection planning
- Resource allocation (where to send documentation teams)
- Verification prioritization
- Funder reporting (what gaps we're addressing)
- Research agenda setting

---

## Cross-Dimensional Synchronization

All four dimensions share:

### Unified Entity IDs
Same entity appears consistently across all outputs:
- `unit_lid99_arty` in Timeline event
- `unit_lid99_arty` in Military Ops Dossier
- `unit_lid99_arty` in Primary Source extraction
- `unit_lid99_arty` in Entity Gap analysis

### Bidirectional Cross-References
From Timeline → Dossier → Archive → back to Timeline:
```json
"cross_references": {
  "timeline_event": "evt_234",
  "dossier_entries": ["dsr_mil_001", "dsr_vio_045"],
  "primary_sources": ["evd_042", "evd_118"],
  "gap_analysis": ["vgap_003"]
}
```

### Shared Provenance
All dimensions trace back to same verified evidence base:
```
Primary Source (evd_042)
  → Extracted Entity (unit_lid99_arty)
    → Verified Cross-Source
      → Timeline Event (evt_234)
      → Dossier Entry (dsr_mil_001)
      → Gap Analysis (vgap_003 - needs corroboration)
```

### Synchronized Updates
When new evidence emerges:
1. Archive dimension adds new primary source
2. Extraction creates new entities
3. Verification checks against existing data
4. Timeline adds/modifies events
5. Dossiers update relevant entries
6. Gap analysis recalculates (some gaps filled, new gaps identified)

All dimensions remain consistent through atomic update transactions.

---

## Output Format Options

Each dimension can export in multiple formats for different tools:

| Dimension | JSON | CSV | PDF Report | GeoJSON | Database |
|-----------|------|-----|------------|---------|----------|
| Timeline | ✓ | ✓ | ✓ | ✗ | ✓ |
| Dossiers | ✓ | ✓ | ✓ | ✗ | ✓ |
| Archives | ✓ | ✗ | ✓ | ✗ | ✓ |
| Gap Analysis | ✓ | ✓ | ✓ | ✓ (geographic gaps) | ✓ |

But all formats derive from **same underlying data model** - format is presentation, not structure.

---

## Why This Works

**Single-output approaches fail because:**
- Timeline format loses thematic organization
- Dossier format loses chronological narrative
- Archive format is too granular for analysis
- Gap analysis is metadata, not primary data

**Multi-dimensional output succeeds because:**
- ✓ Each user gets format optimized for their workflow
- ✓ No information loss through forced compromises
- ✓ Cross-references enable dimension hopping
- ✓ Same evidence base = consistency across views
- ✓ Updates propagate to all relevant dimensions

---

## Implementation via RISE Tools

```bash
# Generate all four dimensions
rise-compose \
  --ingest sources/ \
  --extract full \
  --verify cross-source \
  --output timeline=master_timeline.json \
  --output dossiers=dossiers/ \
  --output archive=evidence_archive/ \
  --output gaps=gap_analysis.json

# Or generate specific dimensions
rise-timeline --verified-events events.json --output timeline.json
rise-dossier --theme military_operations --output mil_ops.pdf
rise-archive --evidence evidence_bundles/ --output archive/
rise-gap --analysis-date 2024-03-31 --output gaps.json
```

---

*Multi-dimensional outputs are not complexity for complexity's sake - they reflect the genuine diversity of investigative needs in conflict documentation. RISE makes it practical to serve all these needs from a single verified evidence base.*
