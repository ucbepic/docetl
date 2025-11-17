# Example Workflow: Viber Screenshot Collection to Forensic Intelligence

This example demonstrates complete RISE pipeline: from raw Viber screenshots (Zawgyi-encoded Burmese text) to verified forensic intelligence outputs (timeline, dossiers, gap analysis, archive).

## Scenario

**Documentation team** in Mae Sot, Thailand receives WhatsApp/V iber screenshots from sources inside Sagaing Region, Myanmar. Screenshots contain:
- Mixed Zawgyi/Unicode Burmese text
- Witness reports of military attacks
- Photos of damage
- Some with faces visible (anonymization needed)
- Varying quality (clear to barely readable)

**Goal**: Transform into forensic-grade intelligence for legal case and human rights reporting.

---

## Input: Raw Evidence Collection

```
viber_screenshots_kale_april_2021/
├── IMG_20210415_001.jpg  (Viber screenshot - Zawgyi text about LID 99 attack)
├── IMG_20210415_002.jpg  (Photo of shell crater)
├── IMG_20210416_001.jpg  (Witness statement - mixed Zawgyi/Unicode)
├── IMG_20210417_001.jpg  (Follow-up report mentioning casualties)
└── ... (146 more files)
```

### Sample Screenshot Content

**IMG_20210415_001.jpg** (Viber message, Zawgyi encoding):
```
[Viber header: Ko Aung, 15/4/2021 16:30]

လိုင်း ၉၉ အမှတ် ၁၄:၀၀ နာရီတွင် ရွာကို ပစ်ခတ်ခဲ့သည်။
အခြောက် ၁၅-၂၀ ချက် ခန့် ပစ်ခတ်ခဲ့သည်။
အိမ် ၃ လုံး ပျက်စီးသွားသည်။
ကျောင်းတိုက်လည်း ပျက်စီးသည်။
ကိုဟောင်း ဒဏ်ရာရရှိသည်။

[Photo attached: shell crater in village]
```

Translation:
> LID 99 shelled the village at 14:00.
> Approximately 15-20 rounds fired.
> 3 houses destroyed.
> Monastery also damaged.
> Ko Aung injured.

---

## Phase 1: Ingestion (rise-ingest)

```bash
rise-ingest \
  --source viber_screenshots_kale_april_2021/ \
  --encoding-hint zawgyi \
  --context myanmar_sagaing \
  --anonymize faces,phone_numbers \
  --embargo-until 2026-01-01 \
  --output evidence_bundles/ \
  --progress-bar
```

### What Happens

1. **File intake**: 150 screenshots processed
2. **Encoding detection**:
   - 112 files detected as Zawgyi (confidence > 0.9)
   - 28 files detected as Unicode
   - 10 files flagged as mixed/uncertain (→ review queue)
3. **OCR**: Tesseract 5.3 with Burmese language pack
4. **Zawgyi → Unicode conversion**: Using `myanmar-tools`
5. **Anonymization**: 23 faces detected and blurred, 15 phone numbers redacted
6. **Provenance**: Chain of custody established for each file

### Output

```
evidence_bundles/
├── bundles/
│   ├── evd_20240110_142300_001.json  (IMG_20210415_001.jpg bundle)
│   ├── evd_20240110_142300_002.json
│   └── ... (150 bundles)
├── originals/
│   ├── evd_20240110_142300_001_original.jpg
│   └── ...
├── review_queue/
│   └── low_encoding_confidence/
│       └── evd_20240110_145000_042.json  (10 items flagged)
└── logs/
    └── ingestion_20240110.log
```

### Sample Evidence Bundle

**evd_20240110_142300_001.json** (simplified):
```json
{
  "evidence_id": "evd_20240110_142300_001",
  "original": {
    "filename": "IMG_20210415_001.jpg",
    "hash_sha256": "a3f7c8e9d2b4f1a6...",
    "size_bytes": 1247893
  },
  "encoding_analysis": {
    "encoding_original": "zawgyi",
    "encoding_confidence": 0.95,
    "zawgyi_indicators": ["'ေ' character before consonant detected"]
  },
  "normalized_content": {
    "text_original_bytes": "[base64 Zawgyi bytes]",
    "text_normalized_unicode": "လိုင်း ၉၉ အမှတ် ၁၄:၀၀ နာရီတွင် ရွာကို ပစ်ခတ်ခဲ့သည်...",
    "ocr_confidence": 0.92
  },
  "security_metadata": {
    "embargo_until": "2026-01-01",
    "anonymization_applied": false
  },
  "quality_assessment": {
    "overall_quality": 0.89,
    "human_review_recommended": false
  },
  "downstream_ready": true
}
```

---

## Phase 2: Extraction (rise-extract)

```bash
rise-extract \
  --input evidence_bundles/bundles/ \
  --entities military_units,locations,persons \
  --events attacks,casualties \
  --context myanmar_tatmadaw_2024 \
  --confidence-threshold 0.7 \
  --output extracted/ \
  --audit-trail extraction_decisions.log
```

### What Happens

1. **Entity extraction**:
   - "လိုင်း ၉၉" → Light Infantry Division 99 (database match, confidence 0.95)
   - "ရွာ" (village) + context → Thayetchaung Village, Kale Township (confidence 0.92)
   - "ကိုဟောင်း" → Ko Aung (person, injured civilian, confidence 0.9)

2. **Event extraction**:
   - Event type: artillery_attack
   - Date: 2021-04-15 (extracted from Viber timestamp)
   - Time: 14:00 (extracted from text)
   - Location: Thayetchaung Village
   - Perpetrator: LID 99
   - Casualty: Ko Aung (injured)
   - Damage: 3 houses destroyed, monastery damaged

3. **Confidence scoring**:
   - ML NER model: 0.88 (high confidence on "လိုင်း ၉၉")
   - Context database match: +0.10 (LID 99 found in database)
   - OCR quality: -0.03 (minor degradation)
   - Final: 0.95

### Output

```
extracted/
├── entities/
│   ├── ent_mil_001.json  (LID 99)
│   ├── ent_loc_001.json  (Thayetchaung Village)
│   ├── ent_per_001.json  (Ko Aung)
│   └── ... (234 entities total)
└── events/
    ├── evt_attack_001.json  (Artillery attack on Thayetchaung)
    └── ... (189 events total)
```

### Sample Extracted Entity

**ent_mil_001.json**:
```json
{
  "entity_id": "ent_mil_001",
  "entity_type": "military_unit",
  "entity_data": {
    "name_burmese": "လိုင်း ၉၉",
    "name_english": "Light Infantry Division 99",
    "unit_type": "light_infantry_division"
  },
  "extraction_provenance": {
    "source_evidence": [
      {
        "evidence_id": "evd_20240110_142300_001",
        "text_span": "လိုင်း ၉၉ အမှတ် ၁၄:၀၀ နာရီတွင် ရွာကို ပစ်ခတ်ခဲ့သည်",
        "character_offsets": [0, 8],
        "extraction_method": "ml_ner_burmese_v2.0 + context_database_match",
        "confidence": 0.95
      }
    ],
    "context_validation": {
      "matched_database_entry": true,
      "database_id": "tatmadaw_lid_99"
    }
  }
}
```

### Sample Extracted Event

**evt_attack_001.json**:
```json
{
  "event_id": "evt_attack_001",
  "event_type": "artillery_attack",
  "event_data": {
    "temporal": {
      "date": "2021-04-15",
      "time": "14:00:00",
      "date_precision": "exact",
      "time_precision": "approximate_hour"
    },
    "spatial": {
      "primary_location": {
        "entity_id": "ent_loc_001",
        "name": "Thayetchaung Village",
        "township": "Kale_Township"
      }
    },
    "entities_involved": {
      "perpetrators": [
        {"entity_id": "ent_mil_001", "role": "attacking_force", "confidence": 0.95}
      ],
      "victims": [
        {"entity_id": "ent_per_001", "role": "injured_civilian", "confidence": 0.9}
      ]
    },
    "incident_details": {
      "weapon_type": "artillery",
      "estimated_rounds": "15-20",
      "casualties": {"killed": 0, "injured": 1},
      "damage": [
        {"type": "residential_building", "count": 3},
        {"type": "monastery", "count": 1}
      ]
    }
  },
  "extraction_provenance": {
    "source_evidence": [{"evidence_id": "evd_20240110_142300_001"}],
    "confidence_breakdown": {
      "overall": 0.88
    }
  }
}
```

---

## Phase 3: Verification (rise-verify)

```bash
rise-verify \
  --entities extracted/entities/ \
  --events extracted/events/ \
  --sources evidence_bundles/bundles/ \
  --cross-source \
  --output verified/ \
  --conflicts conflicts_report.json
```

### What Happens

**Cross-source validation**:
- Event evt_attack_001 mentioned in:
  - evd_001 (primary Viber screenshot)
  - evd_118 (witness statement 2 days later)
  - evd_119 (follow-up report)
- All 3 sources agree on: LID 99, Thayetchaung Village, April 15
- Minor discrepancy: evd_118 says "Ko Aung injured", evd_119 says "3+ injured" (undercount noted)

**Confidence adjustment**:
- Original confidence: 0.88
- Multi-source corroboration: +0.10
- Final confidence: 0.98

**Conflict detection**:
- No major conflicts
- Minor discrepancy flagged in casualties count (1 vs "3+")

### Output

```
verified/
├── entities/
│   └── ent_mil_001_verified.json
├── events/
│   └── evt_attack_001_verified.json
├── conflicts_report.json
└── verification_matrix.csv
```

**conflicts_report.json** (excerpt):
```json
{
  "conflicts": [
    {
      "conflict_id": "conf_001",
      "conflict_type": "casualty_count_discrepancy",
      "severity": "minor",
      "event_id": "evt_attack_001",
      "conflicting_claims": [
        {"source": "evd_001", "claim": "1 injured (Ko Aung)"},
        {"source": "evd_119", "claim": "3+ injured"}
      ],
      "analysis": {
        "likely_resolution": "evd_001 mentions only witness (Ko Aung), evd_119 has fuller count",
        "recommended_action": "Note minimum 1 confirmed (Ko Aung), likely 3+, search for additional casualty reports"
      }
    }
  ]
}
```

---

## Phase 4: Organization - Timeline (rise-timeline)

```bash
rise-timeline \
  --verified-events verified/events/ \
  --identify-gaps \
  --gap-threshold 7 \
  --output timeline.json
```

### Output

**timeline.json** (excerpt):
```json
{
  "timeline_id": "myanmar_sagaing_2021_2024",
  "events": [
    {
      "timeline_entry_id": "tml_234",
      "event_id": "evt_attack_001",
      "sequence_number": 234,
      "temporal": {
        "date": "2021-04-15",
        "time": "14:00:00",
        "myanmar_calendar": "1382_Kason_15"
      },
      "event_summary": {
        "type": "artillery_attack",
        "description": "LID 99 shelled Thayetchaung Village, Kale Township"
      },
      "confidence": {
        "overall": 0.98,
        "sources_count": 3
      },
      "related_events": {
        "previous": "tml_233",
        "next": "tml_235"
      }
    }
  ],
  "temporal_gaps": [
    {
      "gap_id": "gap_001",
      "start_date": "2021-03-15",
      "end_date": "2021-03-22",
      "duration_days": 7,
      "location": "Kale_Township",
      "context": "No documentation, but satellite imagery shows activity",
      "priority": "high"
    }
  ]
}
```

---

## Phase 5: Organization - Dossiers (rise-dossier)

```bash
rise-dossier \
  --themes military_operations,human_rights_violations,displacement \
  --verified-events verified/events/ \
  --verified-entities verified/entities/ \
  --output dossiers/
```

### Output

**dossiers/military_operations.json** (excerpt):
```json
{
  "dossier_id": "dossier_military_ops_sagaing_2021_2024",
  "theme": "military_operations",
  "summary_statistics": {
    "total_incidents": 892,
    "incident_breakdown": {
      "artillery_attacks": 347
    },
    "actors_involved": {
      "perpetrators": [
        {"entity": "LID_99", "incidents": 234}
      ]
    }
  },
  "entries": [
    {
      "entry_id": "dossier_entry_001",
      "incident_type": "artillery_attack",
      "temporal": {
        "date": "2021-04-15"
      },
      "incident_description": {
        "summary": "LID 99 Artillery Unit shelled Thayetchaung Village...",
        "perpetrators": [{"entity_id": "ent_mil_001"}],
        "victims": [{"entity_id": "ent_per_001"}]
      },
      "cross_references": {
        "timeline_entry": "tml_234",
        "related_dossiers": ["human_rights_violations", "displacement"]
      }
    }
  ]
}
```

---

## Phase 6: Gap Analysis (rise-gap)

```bash
rise-gap \
  --timeline timeline.json \
  --verified verified/ \
  --dossiers dossiers/ \
  --prioritize \
  --output gap_analysis.json
```

### Output

**gap_analysis.json** (excerpt):
```json
{
  "priority_gaps": [
    {
      "gap_id": "vgap_001",
      "gap_type": "verification",
      "priority": "high",
      "claim": "3+ injured (only 1 confirmed by name)",
      "collection_recommendations": [
        "Interview Ko Aung for names of other injured",
        "Check hospital records from Kale for April 15-17",
        "Contact local documentation networks for casualty lists"
      ]
    },
    {
      "gap_id": "egap_001",
      "gap_type": "entity",
      "entity_id": "ent_mil_001",
      "missing_information": ["commander_name", "battalion_structure"],
      "priority": "high",
      "collection_recommendations": [
        "Review Tatmadaw documents for LID 99 command in 2021",
        "Interview defected soldiers from LID 99"
      ]
    }
  ]
}
```

---

## Phase 7: Archiving (rise-archive)

```bash
rise-archive \
  --evidence evidence_bundles/ \
  --verified verified/ \
  --timeline timeline.json \
  --dossiers dossiers/ \
  --output forensic_archive/ \
  --encrypt \
  --embargo-enforce \
  --reproducibility-package
```

### Output

```
forensic_archive/
├── originals/  (150 original screenshot files, hash-verified)
├── evidence_bundles/
├── verified/
├── outputs/
│   ├── timeline.json
│   └── dossiers/
├── provenance/
│   └── provenance_ledger.jsonl  (Complete audit trail)
├── security/
│   ├── embargo_registry.json
│   └── access_log.jsonl
└── reproducibility/
    ├── tool_versions.json
    ├── context_databases/
    └── configuration/
```

---

## Full Pipeline (One Command)

```bash
rise-compose \
  --ingest viber_screenshots_kale_april_2021/ \
  --extract entities=all,events=all \
  --verify cross-source \
  --timeline monthly \
  --dossier themes=military_operations,human_rights_violations,displacement \
  --gap-analysis \
  --archive full \
  --context myanmar_sagaing \
  --output kale_analysis_2024/
```

**Result**: Complete forensic intelligence package from raw screenshots.

---

## Using the Outputs

### For Legal Case
```bash
# Generate court-ready chronology PDF
rise-timeline --verified kale_analysis_2024/verified/events/ \
              --format pdf \
              --include-sources \
              --output legal_chronology.pdf

# Generate violations dossier (DOCX for editing)
rise-dossier --theme human_rights_violations \
             --format docx \
             --output violations_dossier.docx
```

### For Research Report
```bash
# Extract statistics
cat kale_analysis_2024/dossiers/military_operations.json | \
    jq '.summary_statistics'

# Output:
# {
#   "total_incidents": 892,
#   "actors_involved": {"perpetrators": [{"entity": "LID_99", "incidents": 234}]},
#   "casualties": {"killed": 345, "injured": 892}
# }
```

### For Gap-Filling Collection
```bash
# Identify high-priority gaps
cat kale_analysis_2024/gap_analysis.json | \
    jq '.priority_gaps[] | select(.priority=="high")'

# Output: Specific collection recommendations
# → Interview Ko Aung for other injured names
# → Check hospital records April 15-17
# → Review Tatmadaw docs for LID 99 commander
```

---

## Summary: Raw to Forensic

**Input**: 150 chaotic Viber screenshots (Zawgyi text, faces visible, mixed quality)

**Process**: Ingest → Extract → Verify → Timeline → Dossiers → Gap Analysis → Archive

**Output**:
- ✅ 150 evidence bundles (originals preserved, Zawgyi→Unicode converted, anonymized, hash-verified)
- ✅ 234 verified entities (military units, locations, persons)
- ✅ 189 verified events (cross-source validated, confidence-scored)
- ✅ Chronological timeline (1247 events, temporal gaps identified)
- ✅ 3 thematic dossiers (military ops, violations, displacement)
- ✅ Gap analysis (23 high-priority collection targets)
- ✅ Forensic archive (complete provenance, reproducibility package)

**Outcome**: Forensic-grade intelligence ready for legal action, human rights reporting, and continued investigation.

---

*From chaos to structured intelligence. This is what RISE systematization achieves.*
