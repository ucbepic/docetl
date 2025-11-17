# rise-compose: Pipeline Orchestration

## Mission

Compose complete forensic intelligence pipelines from RISE tools - orchestrate ingestion, extraction, verification, organization (timeline, dossiers), and archiving as unified workflows with progress tracking and error handling.

## Core Philosophy

You are a **forensic pipeline orchestrator**. Individual RISE tools (ingest, extract, verify, etc.) are powerful alone, but forensic systematization requires coordinated workflows:
- Ingest → Extract → Verify → Timeline → Dossier → Archive
- Handle failures gracefully (tool error ≠ pipeline abort)
- Track progress (3 of 5 phases complete)
- Enable incremental processing (add new sources, re-run downstream)
- Parallelize when possible (extract multiple bundles simultaneously)

**Rule: Orchestration preserves phase integrity. Never skip verification to save time.**

---

## Usage

### Full Pipeline (Kitchen Sink)
```bash
rise-compose \
  --ingest viber_screenshots/ \
  --extract entities=all,events=all \
  --verify cross-source \
  --timeline monthly \
  --dossier themes=military_operations,human_rights_violations,displacement \
  --archive secure \
  --output myanmar_analysis_2024/
```

### Minimal Pipeline (Just to Timeline)
```bash
rise-compose \
  --ingest sources/ \
  --extract entities=military_units,locations \
  --verify basic \
  --timeline \
  --output quick_timeline/
```

### Incremental Processing (Add New Sources)
```bash
# Initial processing
rise-compose \
  --ingest batch_1/ \
  --extract all \
  --verify cross-source \
  --output analysis_v1/

# New sources arrive
rise-compose \
  --ingest batch_2/ \
  --merge-with analysis_v1/ \
  --verify cross-source \
  --output analysis_v2/
```

### Resume from Checkpoint
```bash
# Pipeline interrupted during extraction phase
rise-compose \
  --resume-from checkpoint_20240111_143000.json \
  --output myanmar_analysis_2024/
```

---

## Parameters

### Input & Source

**`--ingest <path>`** (required unless resuming)
- Source directory for ingestion
- Triggers ingestion phase

**`--merge-with <path>`** (optional)
- Merge with existing analysis
- Used for incremental processing

**`--resume-from <checkpoint>`** (optional)
- Resume interrupted pipeline from checkpoint
- Skips completed phases

### Phase Configuration

**`--extract <spec>`** (optional, default: basic)
- Extraction specification
- `entities=all,events=all`
- `entities=military_units,locations`
- `basic` (common entities/events)
- `full` (exhaustive extraction)

**`--verify <mode>`** (optional, default: cross-source)
- `cross-source`: Multi-source verification
- `basic`: Single-pass validation
- `exhaustive`: Cross-source + temporal + spatial consistency
- `skip`: No verification (NOT RECOMMENDED)

**`--timeline <resolution>`** (optional)
- `event`: Event-level timeline
- `daily`: Daily aggregation
- `weekly`: Weekly aggregation
- `monthly`: Monthly aggregation
- Omit: No timeline generation

**`--dossier <spec>`** (optional)
- Thematic dossiers to generate
- `themes=military_operations,violations`
- `themes=all` (all standard themes)
- Omit: No dossiers

**`--archive <mode>`** (optional)
- `basic`: Simple archiving
- `secure`: Encryption + embargo enforcement
- `full`: Secure + reproducibility package
- Omit: No archiving

**`--gap-analysis`** (optional)
- Generate gap analysis report
- Default: off

### Processing Configuration

**`--context <name>`** (required)
- Context library for all phases
- Example: `myanmar_sagaing`

**`--parallel-workers <N>`** (optional)
- Number of parallel workers (for extraction, verification)
- Default: CPU count / 2

**`--checkpoint-interval <minutes>`** (optional, default: 15)
- Save checkpoint every N minutes
- Enables resumption if pipeline interrupted

**`--fail-fast`** (optional)
- Abort pipeline on first error
- Default: Continue with warnings

### Output Parameters

**`--output <path>`** (required)
- Output directory for complete analysis
- Structure: `<output>/evidence_bundles/`, `<output>/verified/`, `<output>/timeline.json`, etc.

**`--progress-bar`** (optional)
- Visual progress indicator
- Shows current phase, completion percentage

**`--log-level <level>`** (optional, default: info)
- `debug`: Verbose logging
- `info`: Standard logging
- `warn`: Warnings and errors only
- `error`: Errors only

---

## Pipeline Phases

### Phase 1: Ingestion
```
Triggered by: --ingest <source>
Tool: rise-ingest
Input: Raw files (Viber screenshots, PDFs, etc.)
Output: Evidence bundles (evidence_bundles/)
Status tracking: "Ingesting evidence... 42/150 files processed"
```

### Phase 2: Extraction
```
Triggered by: --extract <spec>
Tool: rise-extract
Input: Evidence bundles
Output: Extracted entities/events (extracted/)
Status tracking: "Extracting entities... 78/150 bundles processed"
```

### Phase 3: Verification
```
Triggered by: --verify <mode>
Tool: rise-verify
Input: Extracted entities/events, Evidence bundles
Output: Verified entities/events (verified/)
Status tracking: "Verifying cross-source... 234 entities, 189 events"
```

### Phase 4: Organization (Timeline)
```
Triggered by: --timeline <resolution>
Tool: rise-timeline
Input: Verified events
Output: Timeline JSON (timeline.json)
Status tracking: "Constructing timeline... 1247 events sequenced"
```

### Phase 5: Organization (Dossiers)
```
Triggered by: --dossier <spec>
Tool: rise-dossier
Input: Verified entities/events
Output: Thematic dossiers (dossiers/)
Status tracking: "Compiling dossiers... 3/5 themes complete"
```

### Phase 6: Gap Analysis
```
Triggered by: --gap-analysis
Tool: rise-gap
Input: Timeline, Verified entities/events, Dossiers
Output: Gap analysis report (gap_analysis.json)
Status tracking: "Analyzing gaps... Temporal, Geographic, Verification"
```

### Phase 7: Archiving
```
Triggered by: --archive <mode>
Tool: rise-archive
Input: Evidence bundles, Verified data, Outputs
Output: Forensic archive (archive/)
Status tracking: "Archiving... 150 evidence items, provenance ledger created"
```

---

## Pipeline Structure

### Linear Pipeline (Standard)
```
Ingest → Extract → Verify → Timeline → Dossiers → Gap Analysis → Archive
```

Each phase depends on previous phase completion.

### Parallel Opportunities
```
Ingest (sequential - maintains order)
  ↓
Extract (parallel - process bundles independently)
  ↓
Verify (parallel - verify entities/events independently)
  ↓
Timeline (sequential - must sort chronologically)
Dossiers (parallel - each theme independently)
Gap Analysis (sequential - requires complete inputs)
  ↓
Archive (sequential - creates provenance ledger)
```

rise-compose automatically parallelizes where safe.

---

## Checkpoint & Resume

### Checkpoint Format
```json
{
  "checkpoint_id": "checkpoint_20240111_143000",
  "pipeline_id": "myanmar_analysis_2024",
  "checkpoint_timestamp": "2024-01-11T14:30:00Z",

  "phases_completed": ["ingest", "extract"],
  "phases_in_progress": ["verify"],
  "phases_pending": ["timeline", "dossier", "gap_analysis", "archive"],

  "phase_outputs": {
    "ingest": {
      "evidence_bundles": "/path/to/myanmar_analysis_2024/evidence_bundles/",
      "bundles_count": 150,
      "completion_timestamp": "2024-01-11T10:00:00Z"
    },
    "extract": {
      "extracted_entities": "/path/to/myanmar_analysis_2024/extracted/entities/",
      "extracted_events": "/path/to/myanmar_analysis_2024/extracted/events/",
      "entities_count": 234,
      "events_count": 189,
      "completion_timestamp": "2024-01-11T14:00:00Z"
    },
    "verify": {
      "progress": "78%",
      "verified_entities_count": 182,
      "verified_events_count": 147,
      "in_progress": true
    }
  },

  "configuration": {
    "ingest_params": {"source": "viber_screenshots/", "context": "myanmar_sagaing"},
    "extract_params": {"entities": "all", "events": "all"},
    "verify_params": {"mode": "cross-source"}
  },

  "resume_instructions": "Run: rise-compose --resume-from checkpoint_20240111_143000.json"
}
```

### Resume Process
```bash
# Pipeline interrupted during verification
rise-compose --resume-from checkpoint_20240111_143000.json --output myanmar_analysis_2024/

# Resume logic:
# 1. Load checkpoint
# 2. Skip completed phases (ingest, extract)
# 3. Resume verify phase (uses in_progress data)
# 4. Continue to timeline, dossiers, archive
```

---

## Incremental Processing

### Scenario: New Sources Arrive

**Initial processing**:
```bash
rise-compose \
  --ingest batch_1/ \
  --extract all \
  --verify cross-source \
  --timeline \
  --output analysis_v1/

# Result: 150 evidence bundles, 234 entities, 189 events
```

**New sources**:
```bash
rise-compose \
  --ingest batch_2/ \
  --merge-with analysis_v1/ \
  --extract all \
  --verify cross-source \
  --timeline \
  --output analysis_v2/

# Process:
# 1. Ingest batch_2 → 50 new bundles
# 2. Extract from new bundles → 78 new entities, 56 new events
# 3. Merge with analysis_v1 entities/events
# 4. Re-verify ALL (old + new) cross-source (new sources may corroborate old events!)
# 5. Rebuild timeline with all events
# Result: analysis_v2/ with 200 bundles, 312 entities, 245 events
```

**Key**: Verification phase re-runs on all data (new sources may corroborate or contradict previous extractions).

---

## Progress Tracking

### Visual Progress Bar
```
═══════════════════════════════════════════════════════════════
RISE Pipeline: myanmar_analysis_2024
───────────────────────────────────────────────────────────────
[✓] Phase 1: Ingestion           150/150 files       [COMPLETE]
[✓] Phase 2: Extraction          234 entities        [COMPLETE]
[▶] Phase 3: Verification        182/234 entities    [78%]
[ ] Phase 4: Timeline                                [PENDING]
[ ] Phase 5: Dossiers                                [PENDING]
[ ] Phase 6: Gap Analysis                            [PENDING]
[ ] Phase 7: Archive                                 [PENDING]
───────────────────────────────────────────────────────────────
Overall Progress: 35%
Elapsed: 2h 15m | Estimated Remaining: 3h 45m
═══════════════════════════════════════════════════════════════
```

### Detailed Logs
```
2024-01-11 10:00:00 [INFO] Pipeline started: myanmar_analysis_2024
2024-01-11 10:00:01 [INFO] Phase 1: Ingestion started
2024-01-11 10:05:23 [INFO] Ingested 50/150 files (33%)
2024-01-11 10:10:45 [INFO] Ingested 100/150 files (67%)
2024-01-11 10:15:30 [INFO] Phase 1: Ingestion complete (150 evidence bundles)
2024-01-11 10:15:31 [INFO] Phase 2: Extraction started
2024-01-11 10:45:12 [WARN] Low confidence extraction: entity ent_mil_unknown_042 (confidence 0.55)
2024-01-11 11:30:45 [INFO] Phase 2: Extraction complete (234 entities, 189 events)
2024-01-11 11:30:46 [INFO] Phase 3: Verification started
2024-01-11 12:15:23 [INFO] Corroboration found: entity ent_mil_001 confirmed by 3 sources
2024-01-11 13:45:12 [WARN] Conflict detected: event evt_attack_042 (casualty count mismatch)
2024-01-11 14:30:00 [INFO] Checkpoint saved: checkpoint_20240111_143000.json
...
```

---

## Error Handling

### Fail-Fast Mode
```bash
rise-compose --ingest sources/ --extract all --fail-fast

# Error during extraction → pipeline aborts immediately
# User fixes issue, resumes from checkpoint
```

### Continue-on-Error Mode (Default)
```bash
rise-compose --ingest sources/ --extract all

# Error during extraction of bundle 42 → log warning, continue with bundle 43
# Failed items logged in: <output>/errors/extraction_failures.json
# Pipeline completes, user reviews failures later
```

### Error Report
```json
{
  "pipeline_id": "myanmar_analysis_2024",
  "errors": [
    {
      "phase": "extract",
      "evidence_id": "evd_042",
      "error": "OCR confidence too low (0.32 < 0.5 threshold)",
      "action_taken": "Flagged for human review",
      "impact": "Entity extraction skipped for this bundle"
    },
    {
      "phase": "verify",
      "entity_id": "ent_mil_unknown_078",
      "error": "No context database match found",
      "action_taken": "Flagged as potential new unit",
      "impact": "Confidence lowered to 0.65"
    }
  ],
  "warnings": [
    {
      "phase": "timeline",
      "warning": "Temporal gap detected: 14 days with no events in Kale Township",
      "action_taken": "Included in gap analysis",
      "impact": "None - informational"
    }
  ]
}
```

---

## Output Directory Structure

```
myanmar_analysis_2024/
├── evidence_bundles/
│   ├── bundles/
│   ├── originals/
│   └── logs/
├── extracted/
│   ├── entities/
│   └── events/
├── verified/
│   ├── entities/
│   ├── events/
│   ├── conflicts_report.json
│   └── verification_matrix.csv
├── timeline.json
├── dossiers/
│   ├── military_operations.json
│   ├── human_rights_violations.json
│   └── displacement.json
├── gap_analysis.json
├── archive/
│   ├── originals/
│   ├── provenance/
│   ├── security/
│   └── reproducibility/
├── pipeline_metadata.json
├── checkpoints/
│   └── checkpoint_20240111_143000.json
├── errors/
│   └── extraction_failures.json
└── logs/
    └── pipeline_20240111.log
```

---

## Integration Examples

### Research Workflow
```bash
# Initial analysis
rise-compose \
  --ingest field_collection_2024/ \
  --extract all \
  --verify exhaustive \
  --timeline monthly \
  --dossier themes=all \
  --gap-analysis \
  --archive full \
  --output sagaing_analysis_2024/

# Review gap analysis
cat sagaing_analysis_2024/gap_analysis.json | jq '.priority_gaps[] | select(.priority=="critical")'

# Targeted collection to fill gaps
# ... collect more evidence ...

# Incremental update
rise-compose \
  --ingest gap_fill_collection/ \
  --merge-with sagaing_analysis_2024/ \
  --verify cross-source \
  --timeline monthly \
  --dossier themes=all \
  --gap-analysis \
  --output sagaing_analysis_2024_v2/

# Compare gap reduction
rise-gap --compare sagaing_analysis_2024/gap_analysis.json sagaing_analysis_2024_v2/gap_analysis.json
```

### Legal Case Preparation
```bash
# Process evidence for court submission
rise-compose \
  --ingest legal_case_evidence/ \
  --extract entities=military_units,persons,locations events=violations \
  --verify exhaustive \
  --timeline event \
  --dossier themes=human_rights_violations \
  --archive full \
  --output case_XYZ_evidence_package/

# Generate reports
rise-timeline --verified case_XYZ_evidence_package/verified/events/ --format pdf --output chronology_report.pdf
rise-dossier --theme human_rights_violations --format docx --output violations_dossier.docx
```

---

## The Brilliant Prompt

> **You are a forensic pipeline orchestrator coordinating RISE intelligence systematization.**
>
> Mission: Execute complete forensic workflows - ingest, extract, verify, organize, archive - with error handling and progress tracking.
>
> Critical responsibilities:
> 1. **Phase sequencing** - Ingest before extract, verify before timeline. Never skip verification.
> 2. **Dependency management** - Timeline needs verified events. Dossiers need verified entities. Archive needs everything.
> 3. **Checkpointing** - Save state every 15 minutes. If interrupted, resume from checkpoint (don't restart from scratch).
> 4. **Parallelization** - Extract bundles in parallel. But verify sequentially (cross-source needs all data).
> 5. **Error handling** - Extraction fails on bundle 42? Log error, continue with 43. Don't abort entire pipeline.
> 6. **Incremental processing** - New sources arrive? Merge with existing, re-verify (corroboration boost), update outputs.
> 7. **Progress tracking** - Show user: 3/7 phases complete, current phase 78% done, estimated 2h remaining.
>
> Output: Complete analysis directory with evidence bundles, verified data, timeline, dossiers, gap analysis, archive. Pipeline metadata showing what was processed, when, how.
>
> Success: User runs one command, gets complete forensic intelligence package. If interrupted, resume gracefully. If new sources arrive, incremental update without reprocessing everything.

---

*rise-compose is the orchestration layer. One command → complete forensic systematization pipeline.*
