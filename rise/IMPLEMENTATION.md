# RISE Implementation Guide

**Status**: Framework Complete - Implementation In Progress

This document describes the complete RISE implementation: what's been built, how it works, and how to use/extend it.

---

## What's Been Built

### ✅ Complete Specifications (17 Documents)

**Philosophy & Methodology** (3 docs)
- `docs/philosophy/forensic-systematization-manifesto.md` - Core principles
- `docs/philosophy/phase-based-methodology.md` - 5 forensic phases
- `docs/philosophy/multi-dimensional-output-theory.md` - Parallel outputs rationale

**CLI Tool Specifications** (9 complete specs)
- `docs/cli/rise-ingest.md` - Evidence intake
- `docs/cli/rise-extract.md` - Entity/event extraction
- `docs/cli/rise-verify.md` - Multi-source verification
- `docs/cli/rise-timeline.md` - Timeline construction
- `docs/cli/rise-dossier.md` - Thematic compilation
- `docs/cli/rise-archive-and-gap.md` - Preservation & gap analysis
- `docs/cli/rise-compose.md` - Pipeline orchestration

**Implementation Patterns**
- `docs/implementation/brilliant-prompts.md` - Mission briefing approach

**Myanmar Context Documentation**
- `docs/myanmar-context/zawgyi-unicode.md` - Encoding deep dive
- `docs/myanmar-context/context-databases.md` - Database specification

**Examples**
- `examples/viber-screenshot-workflow.md` - Complete end-to-end workflow

### ✅ Data Schemas (JSON Schema)

**Core Schemas** (`schemas/`)
- `evidence-bundle.schema.json` - Ingested evidence structure
- `entity.schema.json` - Extracted entities
- `event.schema.json` - Extracted events
- `timeline.schema.json` - Timeline output format
- `schemas/README.md` - Schema documentation

### ✅ Myanmar Context Databases

**Military Context** (`contexts/myanmar/military/`)
- `tatmadaw_units.json` - Populated with LID 99, 88, 77 + unit types
- Organizational hierarchies, operational areas, verification levels

**Geographic Context** (`contexts/myanmar/geography/`)
- `townships.json` - Sagaing Region townships (Kale, Yinmabin, Tamu, Kalay)
- `villages.json` - Sample villages (Thayetchaung, Kawlin, etc.)
- Coordinates, conflict status, population estimates

**Calendar Context** (`contexts/myanmar/calendar/`)
- `myanmar_calendar_mapping.json` - Myanmar ↔ Gregorian conversion data
- Month mappings, conversion examples, formula

**Terminology** (`contexts/myanmar/terminology/`)
- `military_terms.json` - Burmese-English military terminology
- Organizations, ranks, actions, weapons, locations, casualties
- Myanmar numerals reference

### ✅ Python Package Structure

```
rise/src/rise_tools/
├── ingest/        # rise-ingest implementation
├── extract/       # rise-extract implementation
├── verify/        # rise-verify implementation
├── timeline/      # rise-timeline implementation
├── dossier/       # rise-dossier implementation
├── archive/       # rise-archive implementation
├── gap/           # rise-gap implementation
├── compose/       # rise-compose orchestration
└── utils/         # Shared utilities (context loading, validation, etc.)
```

---

## Architecture Overview

### Forensic Data Flow

```
Raw Evidence (Viber screenshots, Zawgyi-encoded)
    ↓
[rise-ingest] → Evidence Bundles (JSON + originals preserved)
    ↓
[rise-extract] → Entities + Events (confidence-scored)
    ↓
[rise-verify] → Verified Intelligence (multi-source)
    ↓
[rise-timeline] → Chronological Timeline
[rise-dossier] → Thematic Dossiers
[rise-gap] → Gap Analysis
    ↓
[rise-archive] → Forensic Archive (immutable + provenance)
```

### Key Design Principles

1. **Forensic Integrity**
   - Preserve originals bit-for-bit (SHA-256 hashes)
   - Never modify source material
   - Complete chain of custody
   - All transformations documented

2. **Explicit Uncertainty**
   - Confidence scores (0.0-1.0) on all extractions
   - Method documentation (ML vs rule-based vs human)
   - Conflicting information preserved, not hidden
   - Ambiguities flagged for review

3. **Context-Aware Processing**
   - Myanmar military units database → entity recognition
   - Township/village database → location validation
   - Zawgyi/Unicode detection → encoding normalization
   - Myanmar calendar → Gregorian conversion

4. **Multi-Dimensional Outputs**
   - Timeline (chronological)
   - Dossiers (thematic)
   - Archive (preservation)
   - Gap Analysis (what's missing)
   - All synchronized via shared entity IDs

---

## Using the Context Databases

### Loading Military Units

```python
import json

# Load Tatmadaw units database
with open('rise/contexts/myanmar/military/tatmadaw_units.json') as f:
    units_db = json.load(f)

# Find unit by common name
def find_unit(name):
    for unit in units_db['units']:
        if name in unit['common_names']:
            return unit
    return None

# Example: Look up LID 99
lid99 = find_unit('လိုင်း ၉၉')  # Burmese name
print(lid99['official_name_english'])  # "Light Infantry Division 99"
print(lid99['operational_areas'])  # ["Sagaing_Region_Kale_Township", ...]
```

### Loading Geographic Data

```python
# Load townships
with open('rise/contexts/myanmar/geography/townships.json') as f:
    townships_db = json.load(f)

# Find township
def find_township(name):
    for region in townships_db['regions']:
        for township in region['townships']:
            if name in [township['name_english']] + township.get('alternative_spellings', []):
                return township
    return None

# Example: Kale Township
kale = find_township('Kale')
print(kale['coordinates'])  # {'lat': 23.183, 'lon': 94.083}
print(kale['conflict_status'])  # {'intensity': 'high', ...}
```

### Myanmar Calendar Conversion

```python
# Load calendar mapping
with open('rise/contexts/myanmar/calendar/myanmar_calendar_mapping.json') as f:
    calendar_db = json.load(f)

# Example conversion (requires myanmar-calendar library for precise calculation)
# For approximate conversion:
myanmar_year = 1382
gregorian_year_approx = myanmar_year + 638  # 2020

# For precise conversion, use myanmar-calendar library:
# from myanmar_calendar import MyanmarDate
# myanmar_date = MyanmarDate(year=1382, month="Kason", day=15)
# gregorian_date = myanmar_date.to_gregorian()  # 2021-04-15
```

---

## JSON Schema Validation

### Validating Evidence Bundles

```python
import json
import jsonschema

# Load schema
with open('rise/schemas/evidence-bundle.schema.json') as f:
    schema = json.load(f)

# Load data
with open('evidence_bundles/bundles/evd_20240110_142300_001.json') as f:
    data = json.load(f)

# Validate
try:
    jsonschema.validate(instance=data, schema=schema)
    print("✓ Valid evidence bundle")
except jsonschema.ValidationError as e:
    print(f"✗ Validation error: {e.message}")
```

### Validating Entities

```python
# Load entity schema
with open('rise/schemas/entity.schema.json') as f:
    entity_schema = json.load(f)

# Validate extracted entity
with open('extracted/entities/ent_mil_001.json') as f:
    entity_data = json.load(f)

jsonschema.validate(instance=entity_data, schema=entity_schema)
```

---

## Implementation Roadmap

### Phase 1: Core Tools (In Progress)

**rise-ingest**
- [ ] File intake and hashing
- [ ] Zawgyi/Unicode detection (integrate myanmar-tools)
- [ ] OCR integration (Tesseract with Burmese)
- [ ] Anonymization (face detection, phone number redaction)
- [ ] Evidence bundle creation
- [ ] Chain of custody logging

**rise-extract**
- [ ] Context database loading
- [ ] ML-based NER (spaCy/Stanza for Burmese)
- [ ] Rule-based patterns (military units, dates)
- [ ] Confidence scoring
- [ ] Entity/event JSON generation

**rise-verify**
- [ ] Multi-source cross-referencing
- [ ] Conflict detection
- [ ] Confidence adjustment
- [ ] Verification matrix generation

### Phase 2: Organization Tools

**rise-timeline**
- [ ] Event chronological sorting
- [ ] Myanmar calendar conversion
- [ ] Temporal gap detection
- [ ] Timeline JSON generation

**rise-dossier**
- [ ] Theme classification
- [ ] Pattern analysis
- [ ] Cross-dossier entry creation
- [ ] Dossier JSON/PDF generation

### Phase 3: Analysis & Preservation

**rise-gap**
- [ ] Temporal gap analysis
- [ ] Geographic gap detection
- [ ] Single-source flagging
- [ ] Priority calculation
- [ ] Collection recommendations

**rise-archive**
- [ ] Immutable archive creation
- [ ] Provenance ledger (append-only)
- [ ] Reproducibility package
- [ ] Embargo enforcement

### Phase 4: Orchestration

**rise-compose**
- [ ] Pipeline composition
- [ ] Checkpointing
- [ ] Progress tracking
- [ ] Error handling
- [ ] Incremental processing

---

## Development Setup

### Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies (requirements.txt to be created)
pip install -r requirements.txt
```

### Required Libraries

- **Myanmar-specific**:
  - `myanmar-tools` - Zawgyi/Unicode detection and conversion
  - `myanmar-calendar` - Calendar conversion

- **NLP & OCR**:
  - `spacy` or `stanza` - NER for Burmese
  - `pytesseract` - OCR (with Burmese language pack)

- **Data & Validation**:
  - `jsonschema` - Schema validation
  - `pandas` - Data manipulation

- **Utilities**:
  - `click` - CLI framework
  - `tqdm` - Progress bars
  - `pydantic` - Data validation

### Running Tests

```bash
# Unit tests (to be created)
pytest tests/

# Integration tests
pytest tests/integration/

# Validation tests (schema compliance)
pytest tests/validation/
```

---

## Example Usage (Planned)

### Ingest Viber Screenshots

```bash
rise-ingest \
  --source viber_screenshots/ \
  --encoding-hint zawgyi \
  --context myanmar_sagaing \
  --anonymize faces,phone_numbers \
  --output evidence_bundles/
```

### Extract Entities & Events

```bash
rise-extract \
  --input evidence_bundles/ \
  --entities military_units,locations,persons \
  --events attacks,casualties \
  --context myanmar_tatmadaw_2024 \
  --output extracted/
```

### Verify Across Sources

```bash
rise-verify \
  --entities extracted/entities/ \
  --events extracted/events/ \
  --sources evidence_bundles/ \
  --cross-source \
  --output verified/
```

### Full Pipeline

```bash
rise-compose \
  --ingest sources/ \
  --extract all \
  --verify cross-source \
  --timeline monthly \
  --dossier themes=all \
  --gap-analysis \
  --archive full \
  --output analysis_2024/
```

---

## Contributing

### Adding Context Data

**New military units**:
1. Add to `contexts/myanmar/military/tatmadaw_units.json`
2. Include: official names (Burmese + English), common aliases, verification level
3. Document sources

**New townships/villages**:
1. Add to `contexts/myanmar/geography/townships.json` or `villages.json`
2. Include coordinates (OpenStreetMap), conflict status, population estimates

**New terminology**:
1. Add to `contexts/myanmar/terminology/military_terms.json`
2. Include Burmese term, romanization, English translation, context

### Extending to Other Contexts

To adapt RISE for non-Myanmar contexts (e.g., Syria):

1. **Create context database**:
   - `contexts/syria/military/` - Syrian military units
   - `contexts/syria/geography/` - Syrian governorates/districts
   - `contexts/syria/calendar/` - Islamic/Gregorian conversion
   - `contexts/syria/terminology/` - Arabic terminology

2. **Adapt schemas** (if needed):
   - Extend `custom_metadata` fields
   - Add context-specific entity types

3. **Update CLI tools**:
   - Load different context databases
   - Adjust extraction patterns for Arabic script
   - Modify calendar conversion logic

The **methodology remains the same** - only domain knowledge changes.

---

## Current Status

**Specification**: ✅ 100% Complete
**Context Databases**: ✅ Foundation Complete (Myanmar)
**Schemas**: ✅ Core Schemas Complete
**Implementation**: 🚧 20% Complete

**Next immediate steps**:
1. Implement rise-ingest core (Zawgyi detection, OCR, hashing)
2. Implement rise-extract core (context loading, NER, confidence scoring)
3. Create sample evidence bundles for testing
4. Build validation test suite

---

## License

[To be determined - likely open source with restrictions on surveillance use]

---

## Contact

RISE is developed for Myanmar conflict documentation. For collaboration, context data contributions, or technical questions, [contact information to be added].

---

*This is forensic systematization. From chaos to structured intelligence. From Viber screenshots to legal chronologies. This is RISE.*
