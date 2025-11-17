# Phase-Based Methodology

## Overview

RISE processes evidence through five distinct forensic phases. Each phase has:
- **Clear mission**: What this phase accomplishes
- **Specific inputs**: What evidence/data it receives
- **Quality criteria**: How we know it succeeded
- **Defined outputs**: What it produces for the next phase
- **Audit requirements**: What must be documented

This isn't arbitrary - it mirrors how forensic investigators process physical evidence, adapted for digital information in conflict zones.

---

## Phase 1: INGESTION

**Mission**: Securely intake raw evidence, normalize encoding chaos, preserve complete provenance.

### Inputs
- Raw files (any format: Viber screenshots, PDFs, audio, video, handwritten scans, Excel sheets)
- Metadata (source attribution, collection date, collector identity, embargo status)
- Security context (source protection requirements, anonymization needs)

### Processing
1. **Encoding Detection & Normalization**
   - Auto-detect Zawgyi vs Unicode
   - Convert to Unicode for processing while preserving original
   - Flag encoding uncertainties
   - Handle mixed-encoding documents

2. **Format Standardization**
   - OCR for images/scans
   - Transcription for audio/video
   - Extraction for PDFs/documents
   - Structure preservation for spreadsheets

3. **Metadata Enrichment**
   - Generate unique evidence IDs
   - Timestamp ingestion
   - Compute file hashes
   - Document chain of custody

4. **Security Protocols**
   - Apply embargo tagging
   - Anonymize sensitive identifiers
   - Encrypt at rest
   - Log access attempts

### Quality Criteria
- ✓ Original files preserved bit-for-bit
- ✓ All encoding conversions documented
- ✓ Every item has unique provenance ID
- ✓ Security requirements enforced
- ✓ No information loss in format conversion

### Outputs
- **Normalized Evidence Bundle**: Original + normalized versions with provenance metadata
- **Ingestion Log**: Every decision, conversion, and security action taken
- **Quality Report**: Encoding confidence, OCR accuracy, format conversion issues

### CLI Tool
`rise-ingest` - Forensic ingestion pipeline

---

## Phase 2: EXTRACTION

**Mission**: Identify entities, events, relationships, and temporal markers with explicit confidence scoring.

### Inputs
- Normalized Evidence Bundles (from Phase 1)
- Context Libraries (Myanmar military units, townships, ethnic groups, etc.)
- Extraction Rules (domain-specific patterns and terminology)

### Processing
1. **Entity Recognition**
   - Military units (with rank, location, command structure)
   - Locations (villages, townships, regions - referenced against database)
   - Persons (civilians, soldiers, officials - with role identification)
   - Organizations (armed groups, NGOs, government agencies)
   - Burmese terminology preservation

2. **Event Extraction**
   - Incident identification (attacks, arrests, movements, meetings)
   - Action identification (who did what to whom)
   - Temporal markers (dates, times, durations - normalized to Gregorian)
   - Geospatial data (coordinates, place names, relative locations)

3. **Relationship Mapping**
   - Command chains (military hierarchy)
   - Alliances (inter-group connections)
   - Victim-perpetrator links
   - Witness-event connections

4. **Confidence Scoring**
   - Extraction confidence (0.0-1.0)
   - Source reliability assessment
   - Method documentation (rule-based vs ML vs human-verified)
   - Alternative interpretations noted

### Quality Criteria
- ✓ No invented information (low confidence > hallucination)
- ✓ Original Burmese terms preserved alongside translations
- ✓ Every extraction traceable to source location
- ✓ Ambiguities explicitly flagged
- ✓ Context library references documented

### Outputs
- **Structured Entities**: JSON objects with full metadata
- **Extraction Audit Trail**: How each entity/event was identified
- **Confidence Manifest**: Reliability scores with justification
- **Ambiguity Register**: Unresolved questions for human review

### CLI Tool
`rise-extract` - Entity and event extraction with confidence scoring

---

## Phase 3: VERIFICATION

**Mission**: Cross-validate extracted information against multiple sources, detect conflicts, assign final confidence.

### Inputs
- Structured Entities (from Phase 2)
- Multiple Evidence Bundles (for cross-referencing)
- Known Fact Database (pre-verified ground truth)
- Verification Rules (domain-specific validation logic)

### Processing
1. **Multi-Source Cross-Referencing**
   - Match entities across different sources
   - Identify corroborating evidence
   - Detect contradictions
   - Track information evolution over time

2. **Conflict Detection**
   - Date mismatches
   - Location discrepancies
   - Contradictory accounts
   - Mutually exclusive claims

3. **Confidence Adjustment**
   - Boost confidence for corroborated facts
   - Lower confidence for contradicted claims
   - Flag single-source information
   - Document verification logic

4. **Gap Identification**
   - Missing verification sources
   - Unverifiable claims
   - Information needs for resolution

### Quality Criteria
- ✓ Every fact checked against available sources
- ✓ Conflicts documented, not hidden
- ✓ Confidence scores reflect actual evidence strength
- ✓ Single-source facts explicitly marked
- ✓ Verification method is auditable

### Outputs
- **Verified Entities**: Confidence-scored, cross-referenced data
- **Conflict Report**: All contradictions with source citations
- **Verification Matrix**: Which sources confirm/contradict what
- **Single-Source Flags**: Information requiring additional verification

### CLI Tool
`rise-verify` - Multi-source verification and conflict detection

---

## Phase 4: ORGANIZATION

**Mission**: Structure verified information for multiple investigative use cases via parallel output streams.

### Inputs
- Verified Entities (from Phase 3)
- Organizational Rules (temporal, thematic, geographic)
- Output Templates (formats for each output type)

### Processing
1. **Master Timeline Construction**
   - Chronological ordering of all events
   - Date normalization (Myanmar → Gregorian)
   - Temporal gap identification
   - Event clustering (related incidents)

2. **Thematic Dossier Compilation**
   - Military operations dossier
   - Human rights violations dossier
   - Displacement patterns dossier
   - Economic impact dossier
   - Cross-referenced to master timeline

3. **Geographic Mapping**
   - Township-level aggregation
   - Region-level patterns
   - Migration routes
   - Military presence maps

4. **Gap Analysis**
   - Temporal gaps (missing periods)
   - Geographic gaps (uncovered areas)
   - Thematic gaps (under-documented issues)
   - Source gaps (single-source dependencies)

### Quality Criteria
- ✓ Every fact appears in appropriate outputs
- ✓ Cross-references work bidirectionally
- ✓ Gaps are explicit, not implicit
- ✓ Formats serve actual user workflows
- ✓ Outputs remain synchronized

### Outputs
- **Master Timeline**: Comprehensive chronological log
- **Thematic Dossiers**: Topic-organized intelligence
- **Geographic Intelligence**: Location-based analysis
- **Gap Analysis**: Intelligence collection priorities

### CLI Tools
- `rise-timeline` - Temporal analysis and timeline construction
- `rise-dossier` - Thematic dossier compilation
- `rise-gap` - Intelligence gap analysis

---

## Phase 5: PRESERVATION

**Mission**: Archive primary sources with forensic chain of custody, enable future re-analysis.

### Inputs
- Original Evidence (from Phase 1)
- All Processing Artifacts (extractions, verifications, organizations)
- Provenance Chains (complete audit trail)
- Security Metadata (embargo, anonymization records)

### Processing
1. **Primary Source Archiving**
   - Bit-level preservation of originals
   - Redundant storage with integrity checking
   - Format migration planning (long-term accessibility)
   - Metadata preservation

2. **Provenance Documentation**
   - Complete chain of custody
   - Every processing decision logged
   - AI model versions recorded
   - Human interventions documented

3. **Security & Access Control**
   - Embargo enforcement
   - Anonymization verification
   - Access logging
   - Secure retrieval protocols

4. **Re-Analysis Enablement**
   - Processing pipeline versioning
   - Original + all intermediate outputs preserved
   - Context library snapshots
   - Reproducibility documentation

### Quality Criteria
- ✓ Original evidence verifiable (hash-checked)
- ✓ Provenance chain complete and auditable
- ✓ Security requirements maintained
- ✓ Future re-analysis possible
- ✓ Meets forensic evidence standards

### Outputs
- **Forensic Archive**: Immutable primary source repository
- **Provenance Ledger**: Complete processing history
- **Security Audit Log**: Access and protection records
- **Reproducibility Package**: Everything needed to re-run analysis

### CLI Tool
`rise-archive` - Primary source preservation with chain of custody

---

## Pipeline Composition

Individual phases can be chained into complete forensic pipelines:

### Full Pipeline
```bash
rise-compose \
  --ingest viber_screenshots/ \
  --extract entities=military,locations,events \
  --verify cross-source \
  --timeline monthly \
  --dossier themes=violations,operations \
  --archive secure
```

### Targeted Re-Processing
```bash
rise-extract --input evidence_bundle_042.json \
             --focus military_units \
             --context myanmar_tatmadaw_2024.db \
             --confidence-threshold 0.7
```

### Verification-Only
```bash
rise-verify --entities extracted/ \
            --sources evidence_bundles/ \
            --conflicts report \
            --output verification_matrix.json
```

---

## Phase Dependencies

```
INGESTION (foundational - always first)
    ↓
EXTRACTION (requires normalized evidence)
    ↓
VERIFICATION (requires extracted entities)
    ↓
ORGANIZATION (requires verified facts)
    ↓
PRESERVATION (archives everything)
```

**But also:**
- EXTRACTION can be re-run with better context libraries
- VERIFICATION can add new sources to existing analysis
- ORGANIZATION can generate new dossier types
- PRESERVATION is continuous, not a final step

---

## Human-in-the-Loop Integration

Each phase has decision points where human expertise is required:

| Phase | Decision Point | Tool Support |
|-------|---------------|--------------|
| Ingestion | OCR/transcription verification | Confidence-flagged review queue |
| Extraction | Ambiguous entity resolution | Side-by-side original + extraction |
| Verification | Conflict adjudication | Multi-source comparison view |
| Organization | Thematic categorization | Suggested categories + manual add |
| Preservation | Embargo decisions | Security requirement templates |

RISE doesn't eliminate human judgment - it **structures the questions** that require human expertise.

---

## Success Metrics Per Phase

| Phase | Key Metric | Target |
|-------|------------|--------|
| Ingestion | Encoding detection accuracy | >99% (with flagged uncertainties) |
| Extraction | Entity recall (vs human baseline) | >90% (high confidence entities) |
| Verification | Conflict detection rate | 100% (never hide contradictions) |
| Organization | Cross-reference completeness | >95% (relevant facts linked) |
| Preservation | Provenance chain completeness | 100% (every fact traceable) |

---

*This methodology is implemented through the RISE CLI tools. Each tool embodies one or more phases, with shared forensic principles encoded throughout.*
