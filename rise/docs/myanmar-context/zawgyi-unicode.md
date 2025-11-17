# Zawgyi and Unicode: Myanmar's Encoding Crisis

## The Problem

Myanmar has an **encoding crisis** that corrupts data, breaks searches, and creates forensic challenges.

**Two incompatible "standards"** for writing Burmese digitally:
1. **Unicode** (correct, international standard, but adopted late in Myanmar)
2. **Zawgyi** (broken hack-font that looks like Unicode but isn't, widely used)

**Result**: Text that displays correctly in one system becomes gibberish in the other. Documents are mixed. Data is corrupted. Searches fail.

For conflict documentation, this isn't just technical debt - it's **forensic evidence integrity**. Mis-converting Zawgyi destroys evidence.

---

## Technical Background

### Unicode (The Correct Standard)

**Unicode** defines code points for Burmese script (U+1000 to U+109F):
- `U+1000` = ka (က)
- `U+1001` = kha (ခ)
- `U+1031` = vowel sign e (ေ)
- etc.

**Correct rendering** follows Myanmar script rules:
- Vowel diacritics combine with consonants in specific orders
- Medials (subscript forms) attach below consonants
- Tone marks stack above

**Example: "kauN" (ကောင်း = good)**
```
U+1000 (က) + U+102D (ေ) + U102F (ါ) + U+1004 (င်) + U+103A (း)
```

Rendered: ကောင်း

### Zawgyi (The Broken Hack)

**Zawgyi** is a **font hack** created in early 2000s before Unicode support was widespread in Myanmar.

**How it works**:
- Uses same Unicode code points as real Unicode
- But renders them in **wrong positions** via font manipulation
- Relies on specific font file (`Zawgyi-One.ttf`) to display "correctly"
- Without Zawgyi font, displays as gibberish

**Same example: "kauN" in Zawgyi**
```
Different code point sequence (not Unicode-compliant)
Looks like: ေကာင္း (note diacritic positions are different)
```

**The catastrophe**:
- Zawgyi text stored as "Unicode" code points
- But code points don't follow Unicode rules
- If you render with Unicode-compliant font → gibberish
- If you Unicode-normalize or process → corruption

**Analogy**: Imagine two "English" systems:
- Unicode English: "Hello" = H-e-l-l-o
- Zawgyi English: "Hello" = H-l-l-o-e (rearranged)
- Both use same letters (A-Z), but different order
- Font hides the difference, but data processing reveals it

---

## Why Zawgyi Persists

Despite being "wrong", Zawgyi dominates Myanmar digital text:

**Historical**:
- Created 2006, widely adopted before Unicode support
- Became de facto standard
- Millions of devices, documents, websites

**Social**:
- "Zawgyi keyboard" is what people learned
- "Unicode looks wrong" (because they view with Zawgyi font)
- Network effect: If your friends use Zawgyi, you do too

**Technical inertia**:
- Legacy systems built for Zawgyi
- Conversion tools imperfect (lossy, errors)
- Mixed documents (some paragraphs Zawgyi, some Unicode)

**Result**: As of 2021-2024 conflict, **majority of Viber messages, social media, witness statements are Zawgyi**.

Unicode is "winning" slowly, but Zawgyi remains widespread.

---

## Forensic Implications

### Data Corruption Risk

**Scenario**: Witness sends Viber screenshot (Zawgyi text)
```
Original: လိုင်း ၉၉ (LID 99 in Zawgyi)
```

**If you process as Unicode**:
- Text extraction → garbled bytes
- Search for "LID 99" → no match (different byte sequence)
- Entity extraction → fails (ML model trained on Unicode, can't parse Zawgyi)
- Database insertion → corrupted data

**Result**: Evidence lost.

### Mixed Encoding Chaos

**Reality**: Documents are **mixed**:
- Original witness writes in Zawgyi (Viber message)
- Researcher copies to Google Docs (auto-converts to Unicode)
- Export to PDF → Unicode
- But screenshot of original Viber → still Zawgyi

**Same event, two encodings in evidence bundle.**

If not handled: Looks like different information, duplicate detection fails, cross-referencing breaks.

### Search Failures

```
Search for: "လိုင်း ၉၉" (Unicode)
Zawgyi text: လိုင်း ၉၉ (Zawgyi - different bytes!)
Match: NO

Search broken.
```

Can't find evidence even when it exists.

### Legal Evidence Integrity

Court submission requires:
- Original evidence preserved
- Processing documented
- No data corruption

**If Zawgyi → Unicode conversion is undocumented or lossy**:
- Chain of evidence broken
- "Is this the same text as the screenshot?" becomes uncertain
- Legal inadmissibility risk

---

## RISE Detection & Handling

### Detection Strategy

**Multi-method approach**:

1. **ML Classifier**
   - Train on Zawgyi vs Unicode datasets
   - Input: byte sequence
   - Output: zawgyi_probability (0.0-1.0), unicode_probability (0.0-1.0)
   - Tool: `myanmar-tools` (Google's library)

2. **Rule-Based Validation**
   - Check Unicode combination rules
   - Zawgyi violations:
     - `U+1031` (vowel e) BEFORE consonant (violates Unicode rule)
     - Certain code point sequences impossible in Unicode
   - If violations found → likely Zawgyi

3. **Context Clues**
   - File metadata (font name: "Zawgyi-One" → definitely Zawgyi)
   - Source information (Viber screenshot from 2021 → probably Zawgyi)
   - User confirmation (if available)

4. **Confidence Scoring**
   ```python
   ml_score = classifier(text)  # 0.95 = 95% confident Zawgyi
   rule_score = check_unicode_violations(text)  # 0.92 = many violations (Zawgyi indicators)
   context_score = source_metadata_hints(source)  # 0.8 = Viber 2021 (likely Zawgyi)

   final_confidence = weighted_average(ml_score, rule_score, context_score)
   # 0.89 = High confidence Zawgyi

   if final_confidence > 0.85:
       encoding = "zawgyi"
   elif final_confidence < 0.15:
       encoding = "unicode"
   else:
       encoding = "uncertain" → FLAG FOR HUMAN REVIEW
   ```

### Conversion Process

**Tools**: `myanmar-tools`, `Rabbit` converter

**Process**:
```python
# 1. Detect encoding
encoding, confidence = detect_encoding(text)

# 2. If Zawgyi detected:
if encoding == "zawgyi":
    # Preserve original bytes (base64 encoded)
    original_bytes = base64.encode(text.encode('utf-8'))

    # Convert to Unicode
    unicode_text = zawgyi_to_unicode(text)

    # Validate conversion
    is_valid_unicode = check_unicode_rules(unicode_text)

    # Document
    metadata = {
        "encoding_original": "zawgyi",
        "encoding_confidence": confidence,
        "conversion_method": "myanmar-tools v2.1",
        "conversion_timestamp": now(),
        "original_preserved": True,
        "original_bytes": original_bytes,
        "validation_result": is_valid_unicode
    }

    return unicode_text, metadata
```

**Critical: NEVER discard original.** Always preserve Zawgyi bytes before conversion.

### Mixed Encoding Handling

**Problem**: Same document, paragraphs in different encodings.

**Detection**:
- Segment text (by paragraph, sentence, or character runs)
- Detect encoding per segment
- If mixed → flag

**Conversion**:
- Convert segments independently
- Preserve segment boundaries
- Document which segments were which encoding
- Store segment-level metadata

**Example**:
```json
{
  "text_segments": [
    {
      "segment_id": 0,
      "text": "လိုင်း ၉၉ ရဲ့ တိုက်ခိုက်မှု",
      "encoding": "zawgyi",
      "confidence": 0.95,
      "converted_to_unicode": "လိုင်း ၉၉ ရဲ့ တိုက်ခိုက်မှု"
    },
    {
      "segment_id": 1,
      "text": "April 15, 2021 attack on village",
      "encoding": "ascii",
      "confidence": 1.0,
      "converted_to_unicode": "April 15, 2021 attack on village"
    }
  ]
}
```

---

## Validation & Quality Checks

### Post-Conversion Validation

**Check 1: Unicode Rule Compliance**
```python
def is_valid_unicode(text):
    # Check vowel diacritic positions
    # Check medial combinations
    # Check tone mark stacking
    # Unicode-compliant → True
    # Violations → False (conversion error)
```

**Check 2: Round-Trip Test**
```python
zawgyi_text → unicode_text → zawgyi_text_again
if zawgyi_text == zawgyi_text_again:
    conversion_reversible = True  # Good sign
else:
    conversion_lossy = True  # Flag for review
```

**Check 3: Semantic Preservation**
```python
# Human review: Does converted text mean the same thing?
# Show side-by-side:
# - Original (rendered with Zawgyi font)
# - Converted (rendered with Unicode font)
# Human confirms: "Yes, same meaning" or "No, corruption detected"
```

### Flagging for Review

**Low confidence** (0.5-0.8):
- Uncertain encoding
- Mixed encoding detected
- Unusual character combinations

**Conversion failures**:
- Unicode validation fails
- Round-trip test fails
- Semantic check reveals corruption

**Action**: Create review queue entry for human expert.

---

## Human Review Interface

```bash
rise-ingest --review-queue evidence_bundles/review_queue/encoding_uncertain/

# Display:
# 1. Original text (both Zawgyi and Unicode rendering)
# 2. Detected encoding + confidence
# 3. Converted text
# 4. Validation results
#
# Human actions:
# - Confirm Zawgyi → approve conversion
# - Confirm Unicode → mark as Unicode, no conversion needed
# - Unclear → mark as "manual_transcription_required"
```

---

## Long-Term Strategy

### Unicode Transition

**Goal**: Move toward Unicode-only pipeline.

**But**: Must handle legacy Zawgyi data for years to come.

**Approach**:
- Accept Zawgyi input (don't reject)
- Convert to Unicode (with full documentation)
- Process in Unicode (downstream tools)
- Preserve Zawgyi originals (forensic integrity)
- Support Zawgyi search (convert queries too)

### Improved Detection

**Training data**:
- Collect Zawgyi/Unicode pairs (manual annotations)
- Improve ML classifier
- Edge cases (very short text, mixed scripts, Pali/Sanskrit loanwords)

**Context libraries**:
- Common Zawgyi patterns (military terms, place names)
- Unicode equivalents
- Ambiguous cases

---

## Resources

**Tools**:
- `myanmar-tools` (Google) - ML-based detection + conversion
- `Rabbit` - Popular Myanmar converter
- `ICU` (International Components for Unicode) - Normalization

**Datasets**:
- UCSY Myanmar NLP corpus (Unicode)
- myWord (Zawgyi/Unicode parallel corpus)
- Facebook/Meta Myanmar language resources

**References**:
- Unicode Myanmar script specification (U+1000 to U+109F)
- Zawgyi font documentation (reverse-engineering)
- Myanmar Language Commission reports

---

## Summary for RISE Implementation

1. **Detect**: ML classifier + rule validators + context hints → confidence score
2. **Preserve**: Always store original bytes (base64) before any conversion
3. **Convert**: Zawgyi → Unicode using `myanmar-tools`, validate output
4. **Document**: Metadata with encoding, confidence, method, validation results
5. **Flag**: Low confidence (<0.8) or validation failures → human review queue
6. **Search**: Convert search queries too (search both Zawgyi and Unicode versions)

**Never assume.** Never discard. Never hide uncertainty.

**Zawgyi is a forensic challenge, not just a technical annoyance.**

---

*"If the original was Zawgyi, preserve those exact bytes. Convert to Unicode for processing, but keep both. That Zawgyi encoding is evidence too - it tells you when, where, and how this text was created."*
