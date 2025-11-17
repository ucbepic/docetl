# rise-ingest: Forensic Ingestion Pipeline

## Mission

Securely intake raw evidence from conflict zones, normalize encoding chaos (Zawgyi/Unicode), preserve complete provenance, and enforce security protocols - without losing a single bit of original information.

## Core Philosophy

You are a **forensic evidence intake officer**. Your job isn't to clean data - it's to:
- Accept evidence in any state (messy, mixed-encoding, multiple formats)
- Preserve the original bit-for-bit
- Create normalized working copies with full documentation
- Establish chain of custody from moment of ingestion
- Flag security requirements (embargo, anonymization)
- Never guess - when uncertain, flag for human review

**If the original was chaotic, that chaos is evidence too.** Preserve it, document it, but also create structured versions for downstream processing.

---

## Usage

### Basic Ingestion
```bash
rise-ingest \
  --source viber_screenshots/ \
  --output evidence_bundles/ \
  --context myanmar \
  --security-profile standard
```

### With Encoding Specification
```bash
rise-ingest \
  --source mixed_documents/ \
  --encoding-hint zawgyi \
  --encoding-confidence-threshold 0.8 \
  --flag-ambiguous \
  --output evidence_bundles/
```

### Secure Ingestion with Embargo
```bash
rise-ingest \
  --source sensitive_sources/ \
  --output secure_bundles/ \
  --embargo-until 2026-01-01 \
  --anonymize faces,phone_numbers,locations \
  --encrypt-at-rest \
  --access-log ingestion_audit.log
```

### Batch Processing with Progress Tracking
```bash
rise-ingest \
  --source large_collection/ \
  --batch-size 50 \
  --output evidence_bundles/ \
  --progress-bar \
  --resume-from ingestion_checkpoint.json
```

---

## Parameters

### Input Parameters

**`--source <path>`** (required)
- Directory containing raw evidence files
- Accepts any format: images, PDFs, audio, video, documents, spreadsheets
- Can be single file or directory tree
- Preserves original directory structure in metadata

**`--format-hint <type>`** (optional)
- Suggest expected format: `viber_screenshots`, `pdf_reports`, `excel_lists`, `audio_interviews`, `mixed`
- Helps optimize OCR/extraction pipelines
- Default: auto-detect

**`--encoding-hint <encoding>`** (optional)
- Suggest expected text encoding: `zawgyi`, `unicode`, `mixed`, `unknown`
- For Myanmar context, crucial for Zawgyi/Unicode handling
- Default: auto-detect with confidence scoring

**`--encoding-confidence-threshold <float>`** (optional)
- Minimum confidence (0.0-1.0) for automatic encoding detection
- Below threshold → flag for human review
- Default: 0.8 (80% confidence)

### Context Parameters

**`--context <region>`** (required)
- Geographic/cultural context: `myanmar`, `myanmar_sagaing`, `myanmar_rakhine`, etc.
- Loads appropriate:
  - Encoding detection models (Zawgyi for Myanmar)
  - Calendar systems (Myanmar calendar)
  - Language models (Burmese NLP)
  - Geographic databases (township lists)
- Context library must exist in `rise/contexts/<region>/`

**`--language <code>`** (optional)
- Primary language: `my` (Burmese), `en`, `multi`
- Default: inferred from context

### Security Parameters

**`--embargo-until <date>`** (optional)
- Date until which evidence remains restricted
- Format: `YYYY-MM-DD`
- Enforced in all downstream processing

**`--anonymize <elements>`** (optional)
- Comma-separated list: `faces`, `phone_numbers`, `locations`, `names`, `custom_regex`
- Applied during ingestion, original preserved in secure archive
- Logs all anonymization actions

**`--encrypt-at-rest`** (optional)
- Encrypt evidence bundles using AES-256
- Requires key management configuration
- Default: no encryption (plain JSON)

**`--access-log <path>`** (optional)
- Log all access to ingested evidence
- Forensic audit trail
- Format: timestamp, user, action, evidence_id

### Processing Parameters

**`--batch-size <N>`** (optional)
- Process N files at a time
- Useful for large collections
- Default: 100

**`--parallel-workers <N>`** (optional)
- Number of parallel processing workers
- Default: CPU count / 2

**`--resume-from <checkpoint>`** (optional)
- Resume interrupted ingestion from checkpoint file
- Checkpoint saved automatically every batch
- Prevents re-processing already ingested files

**`--flag-ambiguous`** (optional)
- Create human review queue for low-confidence items:
  - Encoding detection < threshold
  - OCR quality < threshold
  - Format conversion issues
- Output: `<output_dir>/review_queue/`

### Output Parameters

**`--output <path>`** (required)
- Directory for evidence bundles
- Structure: `<output>/bundles/<evidence_id>.json`
- Also creates: `<output>/originals/`, `<output>/logs/`

**`--preserve-structure`** (optional)
- Maintain original directory structure in output
- Useful when source organization is meaningful

**`--progress-bar`** (optional)
- Display visual progress indicator
- Shows: files processed, encoding confidence distribution, issues flagged

---

## Output Structure

### Evidence Bundle Format

Each ingested file becomes an **Evidence Bundle** (JSON):

```json
{
  "evidence_id": "evd_20240110_142300_001",
  "bundle_version": "1.0",
  "ingestion_timestamp": "2024-01-10T14:23:00Z",
  "ingestion_tool": "rise-ingest v1.2.0",

  "original": {
    "filename": "IMG_20210415_viber.jpg",
    "path": "viber_screenshots/april_2021/IMG_20210415_viber.jpg",
    "format": "JPEG",
    "size_bytes": 1247893,
    "hash_sha256": "a3f7c8e9d2b4f1a6c5e8d9f2b3a1c4e7d8f9a2b3c4d5e6f7a8b9c0d1e2f3a4b5",
    "created_timestamp": "2021-04-15T16:30:00+06:30",
    "modified_timestamp": "2021-04-15T16:30:00+06:30"
  },

  "source_metadata": {
    "collector": "field_researcher_042",
    "collection_date": "2024-01-10",
    "collection_method": "secure_transfer",
    "source_location": "Sagaing_Region_Kale_Township",
    "source_description": "Viber screenshot from local witness",
    "chain_of_custody": [
      {
        "timestamp": "2021-04-15T16:30:00+06:30",
        "action": "created",
        "actor": "original_source",
        "location": "Kale_Township"
      },
      {
        "timestamp": "2024-01-10T08:00:00Z",
        "action": "collected",
        "actor": "field_researcher_042",
        "method": "encrypted_usb_transfer"
      },
      {
        "timestamp": "2024-01-10T14:23:00Z",
        "action": "ingested",
        "actor": "rise-ingest v1.2.0",
        "system": "forensic_workstation_01"
      }
    ]
  },

  "encoding_analysis": {
    "text_detected": true,
    "primary_language": "my",
    "encoding_original": "zawgyi",
    "encoding_confidence": 0.95,
    "encoding_method": "ml_classifier_myanmar_v2.1",
    "zawgyi_indicators": [
      "Unicode range U+1000-109F with Zawgyi-specific combinations",
      "Detected 'ေ' character before consonant (Zawgyi pattern)"
    ],
    "ambiguous_segments": [
      {
        "text_snippet": "လိုင်း",
        "offset": 45,
        "zawgyi_confidence": 0.95,
        "unicode_confidence": 0.05,
        "flagged_for_review": false
      }
    ]
  },

  "normalized_content": {
    "text_original_bytes": "[base64 encoded original bytes]",
    "text_normalized_unicode": "လိုင်း ၉၉ အမှတ် ၁၄:၀၀ နာရီတွင် ရွာကို ပစ်ခတ်ခဲ့သည်",
    "ocr_applied": true,
    "ocr_engine": "tesseract_5.3_burmese",
    "ocr_confidence": 0.92,
    "text_extraction_issues": [
      {
        "issue": "low_contrast_region",
        "location": "bottom_right",
        "confidence_impact": -0.05
      }
    ]
  },

  "security_metadata": {
    "embargo_until": "2026-01-10",
    "anonymization_applied": true,
    "anonymized_elements": [
      {"type": "face", "location": "bbox[120,45,200,135]", "method": "gaussian_blur_r15"},
      {"type": "phone_number", "pattern": "09xxxxxxxxx", "method": "redaction"}
    ],
    "access_restrictions": "verified_researchers_only",
    "encryption_status": "encrypted_aes256",
    "encryption_key_id": "key_20240110_forensic"
  },

  "format_conversions": [
    {
      "from_format": "JPEG",
      "to_format": "PNG",
      "reason": "lossless_archival",
      "conversion_tool": "imagemagick_7.1",
      "output_path": "evidence_bundles/converted/evd_20240110_142300_001.png"
    }
  ],

  "quality_assessment": {
    "overall_quality": 0.87,
    "encoding_quality": 0.95,
    "ocr_quality": 0.92,
    "format_quality": 0.98,
    "issues": [
      {
        "type": "compression_artifacts",
        "severity": "low",
        "impact": "minor_ocr_degradation"
      }
    ],
    "human_review_recommended": false,
    "review_reason": null
  },

  "downstream_ready": true,
  "next_phase": "extraction"
}
```

### Directory Structure

```
evidence_bundles/
├── bundles/
│   ├── evd_20240110_142300_001.json
│   ├── evd_20240110_142300_002.json
│   └── ...
├── originals/
│   ├── evd_20240110_142300_001_original.jpg
│   ├── evd_20240110_142300_002_original.pdf
│   └── ...
├── converted/
│   ├── evd_20240110_142300_001.png
│   └── ...
├── review_queue/
│   ├── low_encoding_confidence/
│   │   └── evd_20240110_142500_042.json
│   └── ocr_failures/
│       └── evd_20240110_143000_089.json
├── logs/
│   ├── ingestion_20240110.log
│   ├── encoding_decisions.log
│   └── security_actions.log
└── checkpoints/
    └── ingestion_checkpoint_20240110_150000.json
```

---

## Encoding Detection (Myanmar-Specific)

### Zawgyi vs Unicode Detection

**Challenge**: Zawgyi and Unicode both use same Unicode code points but combine them differently, making detection non-trivial.

**RISE Approach**:
1. **ML Classifier** - Trained on Zawgyi/Unicode patterns
2. **Rule-Based Validators** - Check character combination rules
3. **Context Clues** - File metadata, source origin
4. **Confidence Scoring** - Explicit uncertainty quantification

**Detection Pipeline**:
```
Text Input
  ↓
Check for Myanmar script (U+1000-109F)
  ↓
ML classifier → zawgyi_confidence, unicode_confidence
  ↓
Rule validators → constraint_violations
  ↓
Context hints → source_probability
  ↓
Final confidence score
  ↓
If confidence > threshold → auto-convert
If confidence < threshold → flag for human review
```

### Encoding Conversion

**Zawgyi → Unicode**:
- Use `myanmar-tools` library (Google's converter)
- Apply conversion
- Validate output with Unicode rules
- Document conversion in bundle metadata

**Mixed Encoding Handling**:
- Segment-by-segment detection
- Convert segments independently
- Preserve segment boundaries
- Flag potential boundary errors

**Original Preservation**:
- Always keep original bytes (base64 encoded)
- Store both Zawgyi and Unicode versions
- Document detection confidence
- Enable re-conversion if better tools emerge

---

## OCR & Format Extraction

### Image Processing
- **Tesseract 5.3+** with Burmese language pack
- Pre-processing: contrast enhancement, deskewing, noise reduction
- Confidence scoring per word/line
- Layout analysis for structured documents

### PDF Processing
- Text extraction (if text-based PDF)
- OCR (if image-based PDF)
- Metadata extraction (author, creation date, tools used)
- Page-by-page processing with cross-page context

### Audio/Video Processing
- Transcription using speech-to-text (Burmese if available)
- Timestamp alignment
- Speaker diarization (if multiple speakers)
- Store audio waveform for verification

### Spreadsheet Processing
- Preserve cell structure (row/column relationships)
- Extract formulas (not just values)
- Maintain sheet relationships
- Handle merged cells, formatting

---

## Security & Anonymization

### Anonymization Methods

**Face Blurring**:
- Detect faces using ML (OpenCV/dlib)
- Apply Gaussian blur (radius: 15-25px)
- Log bounding boxes
- Verify anonymization quality

**Phone Number Redaction**:
- Pattern matching: Myanmar patterns (`09\d{7,9}`, `\+959\d{7,9}`)
- Redact with `[PHONE_REDACTED]`
- Store pattern in metadata (not actual number)

**Location Anonymization**:
- GPS coordinate fuzzing (reduce precision)
- Village name → Township level
- Configurable anonymization radius

**Name Redaction**:
- NER (Named Entity Recognition) for person names
- Redact or pseudonymize
- Maintain consistency (same person = same pseudonym)

### Embargo Enforcement
- Embed embargo date in every bundle
- Downstream tools check embargo before processing
- Alert if attempting to use embargoed evidence before date
- Audit log of embargo violations

---

## Quality Assurance

### Automatic Checks
- ✓ Original hash matches after ingestion
- ✓ Encoding conversion reversible (Unicode → Zawgyi → Unicode = original)
- ✓ OCR confidence meets threshold
- ✓ Security requirements applied
- ✓ Metadata complete (no null required fields)

### Human Review Queue
Items flagged for review:
- Encoding confidence < 0.8
- OCR confidence < 0.7
- Format conversion errors
- Ambiguous anonymization (faces detected but unclear)
- Unusual file types

### Review Interface
```bash
rise-ingest --review-queue evidence_bundles/review_queue/

# Interactive review:
# - Shows original + detected encoding + conversion
# - Human corrects if wrong
# - Feedback improves ML model
# - Approved items move to main bundle collection
```

---

## Integration with Downstream Phases

### Output for Extraction Phase
```bash
# Ingested bundles ready for extraction
rise-extract \
  --input evidence_bundles/bundles/ \
  --output extracted_entities/ \
  --context myanmar_sagaing
```

### Resumable Processing
```bash
# Ingest phase 1
rise-ingest --source batch_1/ --output bundles/ --checkpoint cp.json

# Ingest phase 2 (adds to existing bundles)
rise-ingest --source batch_2/ --output bundles/ --checkpoint cp.json --resume

# Extraction uses all bundles
rise-extract --input bundles/bundles/ --output entities/
```

---

## Performance & Scalability

### Benchmarks (approximate)
- **Images (OCR)**: ~5-10 per minute (depending on size/quality)
- **PDFs (text extraction)**: ~50-100 per minute
- **PDFs (OCR)**: ~2-5 per minute
- **Audio (transcription)**: ~0.1x realtime (10min audio = 100min processing)

### Optimization Strategies
- Parallel processing (--parallel-workers)
- Batch processing (--batch-size)
- GPU acceleration for OCR/ML (if available)
- Checkpointing for long-running jobs
- Incremental ingestion (don't re-process existing bundles)

---

## Example Workflows

### Workflow 1: Viber Screenshot Collection
```bash
rise-ingest \
  --source viber_screenshots_kale_april_2021/ \
  --format-hint viber_screenshots \
  --encoding-hint zawgyi \
  --context myanmar_sagaing \
  --anonymize faces,phone_numbers \
  --output evidence_bundles/ \
  --progress-bar
```

### Workflow 2: Mixed Document Archive
```bash
rise-ingest \
  --source mixed_archive/ \
  --format-hint mixed \
  --encoding-hint mixed \
  --encoding-confidence-threshold 0.7 \
  --flag-ambiguous \
  --context myanmar \
  --output evidence_bundles/ \
  --batch-size 25 \
  --resume-from checkpoint.json
```

### Workflow 3: Secure Sensitive Sources
```bash
rise-ingest \
  --source highly_sensitive/ \
  --embargo-until 2028-01-01 \
  --anonymize faces,phone_numbers,locations,names \
  --encrypt-at-rest \
  --access-log secure_access.log \
  --output secure_bundles/ \
  --parallel-workers 1  # Single worker for security
```

---

## Troubleshooting

### Issue: Encoding detection failing
**Solution**: Lower confidence threshold, check --encoding-hint, review training data for ML model

### Issue: OCR producing garbage text
**Solution**: Check image quality, try different OCR engines, flag for manual transcription

### Issue: Anonymization missing faces
**Solution**: Review detection threshold, manually verify review queue, adjust blur radius

### Issue: Large collection processing too slow
**Solution**: Increase --parallel-workers, use --batch-size, consider GPU acceleration

---

## The Brilliant Prompt

When implementing `rise-ingest`, the AI model receives this briefing:

> **You are a forensic evidence intake officer processing conflict documentation from Myanmar.**
>
> Your mission: Accept raw evidence files - Viber screenshots, PDFs, photos, anything - and create forensically sound evidence bundles WITHOUT LOSING A SINGLE BIT OF ORIGINAL INFORMATION.
>
> Key responsibilities:
> 1. **Preserve originals bit-for-bit** - compute SHA-256 hash, store original bytes
> 2. **Detect Zawgyi encoding** - Myanmar uses broken Zawgyi font that looks like Unicode but isn't. Auto-detect with ML, but flag ambiguous cases for human review.
> 3. **Normalize to Unicode** - Convert Zawgyi to proper Unicode, but keep both versions
> 4. **Extract text via OCR** - Use Tesseract with Burmese language pack, score confidence
> 5. **Anonymize sensitive elements** - Blur faces, redact phone numbers, BUT LOG WHAT YOU REMOVED
> 6. **Establish chain of custody** - Document every action: who, what, when, where, how
> 7. **Flag uncertainties** - Encoding confidence < 80%? OCR quality low? CREATE REVIEW QUEUE.
>
> You never guess. You never discard information. You never hide uncertainty.
>
> If something is messy, that mess is evidence. Preserve it, document it, create clean versions for analysis - but never pretend the original was clean.
>
> Your output: Evidence bundles (JSON) containing original + normalized + metadata + quality scores + security annotations. Downstream tools trust your bundles to be complete and truthful.
>
> Success = investigator can trust this evidence in court. Failure = information loss or corruption.

This prompt encodes the entire ingestion philosophy into operational context.

---

*rise-ingest is the foundation of the RISE pipeline. Everything downstream depends on this phase doing its job with absolute integrity.*
