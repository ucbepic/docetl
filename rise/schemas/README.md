# RISE JSON Schemas

This directory contains JSON Schema definitions for all RISE data structures. These schemas define the forensic data formats produced and consumed by RISE tools.

## Schema Files

### Core Evidence & Extraction

- **`evidence-bundle.schema.json`** - Evidence bundle from ingestion phase (rise-ingest)
  - Original file metadata + normalized content + provenance + quality assessment
  - Zawgyi/Unicode encoding analysis
  - Security metadata (embargo, anonymization)
  - Chain of custody

- **`entity.schema.json`** - Extracted entity (military unit, person, location, organization)
  - Entity data with Burmese + English names
  - Extraction provenance (source evidence, confidence, method)
  - Context database validation
  - Relationships to other entities

- **`event.schema.json`** - Extracted event (attack, arrest, displacement, etc.)
  - Temporal data (date/time with precision)
  - Spatial data (location with coordinates)
  - Entities involved (perpetrators, victims, witnesses)
  - Incident details (weapons, casualties, damage)
  - Extraction provenance

- **`verified-entity.schema.json`** - Verified entity (post-verification phase)
  - Original entity + verification status
  - Multi-source corroboration analysis
  - Confidence adjustments
  - Conflict detection results

- **`verified-event.schema.json`** - Verified event (post-verification phase)
  - Original event + verification status
  - Cross-source validation
  - Temporal/spatial consistency checks

### Outputs

- **`timeline.schema.json`** - Master timeline output (rise-timeline)
  - Chronologically ordered events
  - Temporal gap analysis
  - Event sequencing (previous/next links)
  - Myanmar calendar + Gregorian dates

- **`dossier.schema.json`** - Thematic dossier (rise-dossier)
  - Theme-organized entries
  - Pattern analysis
  - Summary statistics
  - Cross-references to timeline/archive

- **`gap-analysis.schema.json`** - Intelligence gap analysis (rise-gap)
  - Temporal gaps
  - Geographic gaps
  - Verification gaps (single-source claims)
  - Entity gaps (missing information)
  - Prioritization + collection recommendations

- **`archive-manifest.schema.json`** - Forensic archive manifest (rise-archive)
  - Archive structure
  - Provenance ledger
  - Reproducibility package metadata
  - Access control logs

### Supporting Schemas

- **`provenance.schema.json`** - Provenance chain elements
  - Chain of custody entries
  - Processing history
  - Audit trail format

- **`confidence.schema.json`** - Confidence scoring structures
  - Overall confidence
  - Factor breakdown
  - Adjustment reasoning

- **`conflict.schema.json`** - Conflict/contradiction reporting
  - Conflict types
  - Severity levels
  - Resolution status

## Schema Principles

### 1. Forensic Integrity

Every schema enforces:
- **Provenance**: Track source of every data point
- **Confidence**: Explicit uncertainty quantification
- **Immutability**: Original data preserved (e.g., Zawgyi bytes alongside Unicode)
- **Auditability**: Complete processing history

### 2. Required vs Optional

- **Required fields**: Data essential for forensic validity (evidence_id, timestamps, source attribution)
- **Optional fields**: Enhancements that may not be available for all evidence (GPS coordinates, precise timestamps)

### 3. Enumerations

Controlled vocabularies for:
- Event types (attack, arrest, displacement, etc.)
- Entity types (military_unit, person, location, organization)
- Confidence levels
- Security classifications

Extensible via `"other"` + description field.

### 4. Validation Rules

- **Timestamps**: ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ)
- **IDs**: Structured patterns (evd_YYYYMMDD_HHMMSS_###, ent_TYPE_###, evt_TYPE_###)
- **Hashes**: Lowercase hex SHA-256 (64 characters)
- **Confidence scores**: 0.0-1.0 floats
- **Coordinates**: Standard lat/lon decimals

### 5. Cross-References

Entities and events reference each other via IDs:
- Events reference entities (perpetrators, victims, witnesses)
- Verified data references source evidence (evidence_id)
- Timeline entries reference events (event_id)
- Dossiers reference timeline entries (timeline_entry_id)

All cross-references validated during processing.

## Usage

### Schema Validation (Python)

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
jsonschema.validate(instance=data, schema=schema)
# Raises ValidationError if invalid
```

### Schema Validation (JavaScript/TypeScript)

```typescript
import Ajv from 'ajv';
import evidenceBundleSchema from './rise/schemas/evidence-bundle.schema.json';

const ajv = new Ajv();
const validate = ajv.compile(evidenceBundleSchema);

const data = require('./evidence_bundles/bundles/evd_20240110_142300_001.json');

if (!validate(data)) {
  console.error('Validation errors:', validate.errors);
}
```

### Schema Generation (from examples)

While these schemas are manually crafted for forensic requirements, you can validate examples:

```bash
# Using ajv-cli
npm install -g ajv-cli

ajv validate -s rise/schemas/evidence-bundle.schema.json \
             -d evidence_bundles/bundles/*.json
```

## Schema Versioning

Schemas follow semantic versioning in the `$id` field:
- Major version: Breaking changes (field removal, type change)
- Minor version: Additions (new optional fields)
- Patch version: Documentation updates, clarifications

Current version: **1.0** (initial release)

## Context-Specific Extensions

Base schemas can be extended for specific contexts (Myanmar, Syria, etc.) via `custom_metadata` objects. Extensions should:
- Not override required fields
- Follow same validation principles
- Document context-specific vocabulary

Example Myanmar extension:
```json
{
  "custom_metadata": {
    "myanmar_specific": {
      "myanmar_calendar_date": "1382_Kason_15",
      "tatmadaw_unit_code": "LID_99_A_1",
      "township_code": "MMR_SAG_KAL_001"
    }
  }
}
```

## Contributing

When modifying schemas:
1. Update schema file
2. Update this README
3. Add validation tests
4. Update relevant CLI tool documentation
5. Increment version number in `$id`

## License

[To be determined - same as RISE project license]

---

*These schemas are the structured foundation of RISE forensic systematization. Everything flows through these formats.*
