# RISE: Rapid Intelligence Systematization Engine

**Forensic systematization as methodology for conflict documentation.**

---

## What Is RISE?

RISE is not a data pipeline. It is a **forensic methodology encoded into software** - a systematic approach to transforming chaotic, dangerous, unreliable information from conflict zones into structured intelligence that can withstand scrutiny.

Born from Myanmar conflict documentation, RISE addresses the gap between:
- **What conflict documentation reality gives us**: Mixed-format Viber screenshots in Zawgyi encoding with uncertain dates and single-source claims
- **What traditional data tools expect**: Clean CSV files in UTF-8 with validated schemas

RISE bridges this gap through **forensic integrity**: treating raw information as evidence that must be preserved, verified, and organized without losing uncertainty or provenance.

---

## The Core Problem

You're documenting human rights violations in an active conflict zone. Your "data" is:

- 📱 Viber screenshots (half in broken Zawgyi encoding, half in Unicode, many mixed)
- 📄 Handwritten notes in Burmese using Myanmar calendar dates
- 🎙️ Voice recordings from witnesses who saw "လိုင်း ၉၉" shell their village
- 📊 Excel spreadsheets tracking displaced families with merged cells and formatting as meaning
- 🗺️ Location references like "the village near the burned monastery" (no coordinates)

Your investigators need:
- ⚖️ Legal chronologies for court submissions (exact dates, complete provenance)
- 📚 Thematic dossiers (all military operations, all displacement events)
- 🔬 Primary source archives (original evidence with chain of custody)
- 🎯 Gap analysis (what we don't know, where to investigate next)

**Traditional ETL pipelines fail**. They assume clean data, reliable sources, single formats, Western dates. They collapse uncertainty, discard originals, hide conflicts between sources.

**RISE succeeds** by treating documentation as **forensic bookkeeping**: preserve everything, verify everything, organize for multiple uses, make gaps explicit.

---

## The RISE Approach

### 1. Forensic Integrity Over Convenience

**Never modify source material.** Never discard uncertainty. Never collapse provenance for simplicity.

```
Traditional: Zawgyi text → auto-convert to Unicode → discard original
RISE: Zawgyi text → preserve original bytes → convert to Unicode → document conversion → keep both
```

If the original was messy, that mess is **evidence too**.

### 2. Multi-Dimensional Output

One output format cannot serve:
- Investigators needing chronological timelines
- Researchers needing thematic dossiers
- Verification teams needing primary source access
- Analysts needing gap identification

**RISE generates four parallel outputs** from the same verified evidence base:
1. **Master Timeline** - Chronological event log
2. **Thematic Dossiers** - Topic-organized intelligence
3. **Primary Archives** - Original evidence with chain of custody
4. **Gap Analysis** - Explicit identification of missing information

All synchronized through shared entity IDs and provenance chains.

### 3. Explicit Uncertainty

Every extracted fact carries:
- Confidence score (0.0-1.0)
- Source attribution
- Extraction method
- Conflicting information (if any)
- Known gaps

**Intelligence value = knowing what we don't know as much as what we do.**

### 4. Phase-Based Processing

Evidence flows through distinct forensic phases:

```
1. INGESTION  → Secure intake, encoding normalization, metadata preservation
2. EXTRACTION → Entity/event identification with confidence scoring
3. VERIFICATION → Cross-source validation, conflict detection
4. ORGANIZATION → Timeline, dossiers, gap analysis
5. PRESERVATION → Primary source archiving, provenance documentation
```

Each phase has clear quality criteria and audit requirements.

### 5. Context as Infrastructure

Myanmar-specific knowledge isn't a plugin - it's **foundational**:
- Zawgyi/Unicode handling prevents data corruption
- Myanmar calendar conversion makes dates meaningful
- Military unit databases enable entity recognition
- Township hierarchies provide geographic structure

Context deeply embedded in every processing stage.

---

## How RISE Works

### The CLI Tools

RISE is a suite of **forensic intelligence tools**, each embodying one phase:

| Tool | Phase | Mission |
|------|-------|---------|
| `rise-ingest` | Ingestion | Secure intake, encoding normalization, provenance establishment |
| `rise-extract` | Extraction | Entity/event extraction with confidence scoring |
| `rise-verify` | Verification | Multi-source cross-validation, conflict detection |
| `rise-timeline` | Organization | Chronological event timeline construction |
| `rise-dossier` | Organization | Thematic intelligence compilation |
| `rise-archive` | Preservation | Primary source archiving with chain of custody |
| `rise-gap` | Analysis | Intelligence gap identification and prioritization |
| `rise-compose` | Orchestration | Pipeline composition and workflow management |

### Example Workflow

```bash
# Phase 1: Ingest Viber screenshots with Zawgyi handling
rise-ingest \
  --source viber_screenshots_april_2021/ \
  --encoding-hint zawgyi \
  --context myanmar_sagaing \
  --anonymize faces,phone_numbers \
  --output evidence_bundles/

# Phase 2: Extract entities and events
rise-extract \
  --input evidence_bundles/ \
  --entities military_units,locations,persons \
  --events attacks,arrests,displacements \
  --context myanmar_tatmadaw_2024 \
  --confidence-threshold 0.7 \
  --output extracted/

# Phase 3: Verify across sources
rise-verify \
  --entities extracted/entities/ \
  --events extracted/events/ \
  --sources evidence_bundles/ \
  --cross-source \
  --conflicts conflicts_report.json \
  --output verified/

# Phase 4: Generate outputs
rise-timeline --verified-events verified/events/ --output timeline.json
rise-dossier --theme military_operations --output dossiers/mil_ops.pdf
rise-gap --analysis-date 2024-03-31 --output gap_analysis.json

# Phase 5: Archive with provenance
rise-archive --evidence evidence_bundles/ --verified verified/ --output forensic_archive/
```

Or compose the entire pipeline:

```bash
rise-compose \
  --ingest viber_screenshots/ \
  --extract entities=all,events=all \
  --verify cross-source \
  --timeline monthly \
  --dossier themes=violations,operations,displacement \
  --archive secure \
  --output rise_analysis_2024/
```

---

## Key Features

### Encoding Intelligence (Zawgyi/Unicode)

Myanmar's "encoding chaos" isn't just technical debt - it's **forensically significant**:
- Zawgyi (broken font masquerading as Unicode) used by millions
- Auto-detection using ML + rule-based validation
- Confidence scoring for encoding decisions
- **Preservation of original bytes** + normalized Unicode
- Mixed-encoding document handling (segment-by-segment)

### Confidence Scoring

Every extraction has explicit uncertainty:
```json
{
  "entity": "Light Infantry Division 99",
  "confidence": 0.95,
  "confidence_factors": [
    {"factor": "ml_extraction", "score": 0.88},
    {"factor": "context_db_match", "boost": +0.10},
    {"factor": "multi_source_corroboration", "boost": +0.05},
    {"factor": "minor_date_conflict", "penalty": -0.03}
  ],
  "sources": ["evd_042", "evd_118", "evd_119"],
  "method": "hybrid_ml_rules"
}
```

Low confidence with documented uncertainty > confident hallucination.

### Conflict Detection

**100% conflict detection rate** - never hide contradictions:
```json
{
  "conflict_type": "casualty_count_mismatch",
  "severity": "major",
  "source_A": {"claim": "3 killed", "confidence": 0.9},
  "source_B": {"claim": "15 killed", "confidence": 0.85},
  "resolution": "requires_human_review",
  "recommended_action": "Search for follow-up reports, interview additional witnesses"
}
```

### Provenance Chains

Every fact traceable to source:
```
Primary Source (evd_042: Viber screenshot)
  → Encoding: Zawgyi detected (0.95 confidence), converted to Unicode
  → Extracted Entity: "LID 99" (0.95 confidence via ML + context match)
  → Verified: Corroborated by evd_118, evd_119 (confidence → 0.98)
  → Timeline Event: evt_234 (2021-04-15, Artillery Attack)
  → Dossier Entry: Military Operations dossier, entry #47
  → Gap Analysis: "LID 99 commander name unknown - high priority"
```

Full audit trail from screenshot to intelligence product.

### Multi-Calendar Support

Myanmar calendar → Gregorian conversion:
```
Burmese: ၁၃၈၂ ခုနှစ် ကဆုန်လ ၁၅ ရက်
Myanmar Calendar: 1382 Kason 15
Gregorian: 2021-04-15
Confidence: 0.95 (standard calendar conversion)
```

Both versions preserved in metadata.

---

## Documentation Structure

### Philosophy Documents
Core principles guiding RISE development:
- [`forensic-systematization-manifesto.md`](docs/philosophy/forensic-systematization-manifesto.md) - What RISE is and why it exists
- [`phase-based-methodology.md`](docs/philosophy/phase-based-methodology.md) - The five forensic phases
- [`multi-dimensional-output-theory.md`](docs/philosophy/multi-dimensional-output-theory.md) - Why four parallel outputs

### CLI Tool Specifications
Detailed operational guides for each tool:
- [`rise-ingest.md`](docs/cli/rise-ingest.md) - Forensic ingestion pipeline
- [`rise-extract.md`](docs/cli/rise-extract.md) - Entity/event extraction
- [`rise-verify.md`](docs/cli/rise-verify.md) - Multi-source verification
- [More tools documented in `docs/cli/`](docs/cli/)

### Technical Architecture
Implementation patterns and schemas:
- JSON schemas for evidence bundles, entities, events
- Pipeline composition patterns
- Provenance chain specifications
- Uncertainty preservation methods

### Myanmar Context Library
Domain-specific knowledge resources:
- Zawgyi/Unicode deep technical dive
- Myanmar calendar conversion rules
- Military unit recognition patterns
- Township/village database structure
- Cultural context and terminology

### Implementation Patterns
How to build RISE tools:
- The "brilliant prompt" approach (mission briefings, not pseudocode)
- Progress visualization patterns
- Human-in-the-loop decision points
- Forensic audit trail requirements

---

## Who Is RISE For?

### Primary Users
- **Human rights documentation teams** processing evidence from conflict zones
- **Forensic investigators** building legal cases requiring evidence integrity
- **Research organizations** analyzing complex events with uncertain information
- **Archivists** preserving conflict documentation for long-term access

### Use Cases
- ✅ Myanmar conflict documentation (primary use case)
- ✅ Multi-language evidence processing with encoding challenges
- ✅ Cross-source verification under uncertainty
- ✅ Legal-grade evidence chain of custody
- ✅ Intelligence gap analysis for collection planning
- ✅ Any scenario where **forensic integrity > processing speed**

### NOT For
- ❌ Clean corporate data with reliable schemas
- ❌ Real-time streaming analysis
- ❌ Cases where source modification is acceptable
- ❌ Contexts where hiding uncertainty is preferred

---

## Design Principles

### The "Brilliant Prompt" Approach

RISE tools are not configured with YAML or pseudocode. They are **briefed like forensic investigators**:

> "You are a forensic evidence intake officer processing Viber screenshots from Sagaing Region. Your mission: extract every military movement mentioned, preserve exact Burmese terminology, note any Zawgyi encoding, cross-reference with township databases, flag temporal ambiguities, and create an audit trail showing your extraction decisions. Never guess - when uncertain, flag for human review."

This approach:
- ✅ Encodes domain knowledge into operational context
- ✅ Makes AI models active participants in methodology
- ✅ Preserves interpretive reasoning for audit
- ✅ Adapts to source-specific challenges

### Human-in-the-Loop, Not Human-Out-of-Loop

RISE doesn't eliminate human judgment - it **structures the questions** requiring human expertise:

| Phase | Decision Point | Tool Support |
|-------|---------------|--------------|
| Ingestion | OCR verification | Confidence-flagged review queue |
| Extraction | Ambiguous entities | Side-by-side original + extraction |
| Verification | Conflict adjudication | Multi-source comparison view |
| Organization | Thematic categorization | Suggested categories + manual override |

### Explicit Over Implicit

```
BAD (implicit):  "3 people killed" (no source, no confidence, no conflicts noted)
GOOD (explicit): "3 killed [source: evd_042, confidence: 0.8, contradicted by evd_150 (15 killed), requires_review]"
```

Make every assumption, uncertainty, and decision **visible** in the output.

---

## Success Criteria

RISE succeeds when:
- ✅ **Investigators trust the data** enough to base legal action on it
- ✅ **Source communities feel represented** accurately and safely
- ✅ **Analysts can answer** "what happened when where why"
- ✅ **Gaps are explicit** so resources can be targeted
- ✅ **Methods are auditable** by external forensic experts
- ✅ **Primary sources are preserved** for future re-analysis

RISE fails if it:
- ❌ Corrupts original evidence
- ❌ Hides uncertainty or conflicts
- ❌ Loses provenance chains
- ❌ Serves only one use case (timeline OR dossier, not both)
- ❌ Requires PhDs to operate
- ❌ Takes longer than manual methods without quality improvement

---

## Beyond Myanmar

While built for Myanmar conflict documentation, RISE methodology applies to:
- 🌍 Human rights investigations in any contested environment
- 📚 Historical archive digitization with complex encodings (non-Latin scripts, legacy formats)
- 🔍 Multi-source intelligence fusion under uncertainty (OSINT, investigative journalism)
- ⚖️ Legal evidence management requiring forensic-grade provenance
- 🏛️ Cultural heritage documentation in crisis contexts

**Any scenario where forensic integrity matters more than processing speed.**

---

## Current Status

**RISE is in active design and development.**

This repository contains:
- ✅ Complete philosophical foundation and methodology
- ✅ CLI tool specifications (rise-ingest, rise-extract, rise-verify, etc.)
- ✅ JSON schemas for forensic data types
- ✅ Myanmar context library framework
- 🚧 Implementation (in progress)
- 🚧 Example pipelines
- 📅 Deployment tools (planned)

**Current focus**: Building the core forensic ingestion and extraction pipeline for Myanmar Viber screenshot collections.

---

## Contributing

RISE is developed with the Myanmar conflict documentation community as primary stakeholder. Contributions welcome in:

- **Context libraries**: Myanmar military unit databases, township lists, terminology
- **Encoding tools**: Improved Zawgyi/Unicode detection, other script challenges
- **Verification methods**: Cross-source validation algorithms
- **Use case testing**: Apply RISE to other conflict documentation scenarios
- **Audit**: Review methodology for forensic soundness

---

## The Vision

A world where small documentation teams can achieve **forensic-grade systematization** without enterprise budgets.

Where the choice isn't between "fast and sloppy" or "perfect and impossible."

Where documentation from dangerous environments gets the technical rigor it deserves.

**RISE is forensic systematization as methodology.**

**RISE is chaos into structured intelligence.**

**RISE is the bookkeeper that conflict documentation needs.**

---

## License

[To be determined - likely open source with usage restrictions for commercial surveillance applications]

---

## Contact

RISE is developed as part of Myanmar conflict documentation efforts. For collaboration or questions:

[Contact information to be added]

---

*"If the original was messy, that mess is evidence too. Preserve it, document it, create clean versions for analysis - but never pretend the original was clean."*

*— The Forensic Systematization Manifesto*
