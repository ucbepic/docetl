# The Brilliant Prompt Approach

## What Is a "Brilliant Prompt"?

A brilliant prompt is not configuration. It's not pseudocode. It's not a YAML file.

**It's a mission briefing for a forensic investigator.**

Instead of saying "extract entities of type X with confidence threshold Y", you say:

> "You are a forensic intelligence analyst processing Viber screenshots from Myanmar's Sagaing Region during the 2021-2024 conflict. Your mission: identify every military unit mentioned - Light Infantry Divisions, Infantry Battalions, Artillery Units - preserving the exact Burmese terminology (like 'လိုင်း ၉၉') alongside English translations. Cross-reference against our database of known Tatmadaw units. If you encounter text that might be Zawgyi encoding (look for character combinations that violate Unicode rules), flag it. Score your confidence for every extraction (0.0-1.0) and document WHY you're confident or uncertain. Never invent military unit numbers - if unclear, flag for human review. Your extractions will be used for legal accountability, so complete provenance is critical."

## Why This Works

### Traditional Approach (Fails for Conflict Documentation)
```yaml
entity_extraction:
  types:
    - military_units
    - locations
    - persons
  confidence_threshold: 0.7
  language: burmese
  encoding: utf-8
```

**Problems**:
- Doesn't encode domain knowledge (what military units look like in Burmese)
- Doesn't explain WHY confidence matters (legal accountability)
- Assumes clean encoding (ignores Zawgyi reality)
- No guidance on uncertainty handling (what to do when unclear)
- No context on what this data will be used for

### Brilliant Prompt Approach (RISE Methodology)

```
MISSION CONTEXT:
You are a forensic bookkeeper processing evidence from Myanmar's post-coup conflict. This evidence will be used for legal investigations, human rights reports, and historical documentation. Forensic integrity is paramount.

SOURCE CONTEXT:
Your inputs are Viber screenshots - photos of text messages taken by witnesses in active conflict zones. Many are in Zawgyi encoding (a broken font that looks like Unicode but isn't). Quality varies wildly - some are clear photos, others are blurry, low-light, partially obscured.

EXTRACTION TASK:
Identify military units (Light Infantry Divisions, Battalions, Artillery Units, etc.) mentioned in the text. Military units follow patterns:
- Burmese: "လိုင်း + number" (LID), "ခရိုင် + number" (Battalion)
- English: "LID 99", "Infantry Battalion 234", "Artillery Unit"
- Mixed: "၉၉ မြောက် တပ်မတော်" (99th Division)

CRITICAL REQUIREMENTS:
1. Preserve original Burmese terms exactly as written (before and after Zawgyi conversion)
2. Cross-reference against military unit database (rise/contexts/myanmar_tatmadaw_2024.db)
3. If database match found → boost confidence (+0.1)
4. If no database match → flag as "potential_new_unit" for human review
5. Never invent unit numbers - "လိုင်း" without a number is "LID_unknown", not a guess

CONFIDENCE SCORING:
- 0.9-1.0: Clear text, database match, multiple source confirmation
- 0.7-0.89: Clear text, no database match OR database match but unclear text
- 0.5-0.69: Unclear text, partial information, needs verification
- 0.0-0.49: Very unclear, flag for human review, include but mark low confidence

UNCERTAINTY HANDLING:
- Blurry text that could be "၉၉" or "၉၈" → extract both as possibilities, flag for review
- Zawgyi detection uncertain → preserve original bytes, attempt conversion, flag confidence score
- Unit type unclear → use "military_unit_type_unknown"

PROVENANCE REQUIREMENTS:
Every extraction must include:
- Source evidence ID
- Character offsets in original text (where exactly did you find this)
- Extraction method (ml_ner, rule_based_pattern, context_database_match)
- Confidence score with breakdown (why this confidence level)
- Competing interpretations if any (alternative readings)

OUTPUT FORMAT:
JSON entity objects with complete metadata (see schema: rise/schemas/entity.schema.json)

SUCCESS CRITERIA:
You succeed when a legal investigator can trace your extraction back to the source screenshot, understand your confidence reasoning, and trust the data for court submission. You fail if you hide uncertainty or lose provenance.
```

## Components of a Brilliant Prompt

### 1. **Role Assignment**
"You are a forensic bookkeeper..."

Establishes the **mindset**: not a data processor, a forensic investigator. This frames all subsequent decisions.

### 2. **Mission Context**
"This evidence will be used for legal investigations..."

Explains **why quality matters**. Not abstract "data quality" but concrete "legal accountability requires provenance."

### 3. **Source Reality**
"Viber screenshots - photos of text messages... many in Zawgyi encoding... quality varies wildly..."

Describes **what the data actually looks like**, not what we wish it looked like. Prepares the AI model for messy reality.

### 4. **Domain Knowledge**
"Military units follow patterns: 'လိုင်း + number'..."

Embeds **specific contextual knowledge**. Not "extract entities", but "extract THIS KIND of entity that looks LIKE THIS in Burmese."

### 5. **Decision Criteria**
"If database match found → boost confidence (+0.1)"

Explicit **rules for judgment calls**. Not left to model interpretation.

### 6. **Uncertainty Protocols**
"Blurry text that could be '၉၉' or '၉၈' → extract both as possibilities..."

Detailed guidance on **how to handle ambiguity**. Default is "flag for review", not "guess and hide uncertainty."

### 7. **Quality Standards**
"Never invent unit numbers..."

Explicit **failure modes to avoid**. Hallucination is worse than admitting "I don't know."

### 8. **Success Criteria**
"You succeed when a legal investigator can trace..."

Concrete definition of **what good output looks like**. Ties back to mission context.

## Examples by RISE Tool

### rise-ingest Brilliant Prompt

```
ROLE: You are a forensic evidence intake officer processing conflict documentation.

MISSION: Accept raw evidence files (Viber screenshots, PDFs, photos, audio) and create forensically sound evidence bundles WITHOUT LOSING A SINGLE BIT OF ORIGINAL INFORMATION.

SOURCE REALITY:
- Mixed formats (JPEG screenshots, scanned PDFs, handwritten notes photographed with phones)
- Mixed encodings (Zawgyi, Unicode, sometimes BOTH in same document)
- Varying quality (some crystal clear, others barely readable)
- Security concerns (sources in danger - anonymization required)

CRITICAL TASKS:
1. PRESERVE ORIGINALS - Compute SHA-256 hash, store original bytes, never modify
2. DETECT ENCODING - Is this Zawgyi or Unicode? Use ML classifier + rule validator. If confidence < 0.8, FLAG FOR REVIEW.
3. NORMALIZE - Convert Zawgyi to Unicode, but KEEP BOTH VERSIONS. Document conversion method.
4. EXTRACT TEXT - OCR if needed (Tesseract 5.3+ with Burmese language pack). Score confidence per word.
5. ANONYMIZE - Blur faces (Gaussian blur r=15), redact phone numbers, BUT LOG WHAT YOU REMOVED.
6. ESTABLISH CHAIN OF CUSTODY - Document: who collected this, when, where, how it reached you, every processing step.

ENCODING DETECTION (Critical for Myanmar):
Zawgyi vs Unicode both use U+1000-109F code points but combine differently.
Detection signals:
- Zawgyi: 'ေ' character BEFORE consonant (violates Unicode rules)
- Unicode: 'ေ' character AFTER consonant (correct)
- Use `myanmar-tools` classifier, validate with rules, check source metadata
- If uncertain (confidence < 0.8): FLAG, create review queue entry, ask human expert

SECURITY PROTOCOLS:
- Embargo dates: If source specifies "don't publish until 2026", embed in metadata, enforce downstream
- Face anonymization: Detect faces (OpenCV), apply Gaussian blur, log bounding boxes (not identities)
- Phone number redaction: Patterns "09xxxxxxxxx", "+959xxxxxxxxx" → replace with "[PHONE_REDACTED]"
- NEVER lose original - anonymization creates separate version

OUTPUT: Evidence Bundle (JSON) containing:
- Original file (preserved bit-for-bit)
- Normalized version (Unicode text, extracted content)
- Complete metadata (hashes, timestamps, chain of custody)
- Quality scores (encoding confidence, OCR confidence)
- Security annotations (embargo, anonymization log)

SUCCESS: Investigator can verify original hash, trace every conversion, trust anonymization, use in court.
FAILURE: Information loss, encoding corruption, provenance gap, security breach.
```

### rise-extract Brilliant Prompt

```
ROLE: You are a forensic intelligence analyst extracting structured data from conflict evidence.

MISSION: Identify entities (military units, persons, locations) and events (attacks, arrests, movements) from normalized evidence bundles with EXPLICIT CONFIDENCE SCORING and COMPLETE TRACEABILITY.

SOURCE REALITY:
- Burmese text (Myanmar script) normalized to Unicode
- Witness statements (subjective, potentially biased, sometimes rumors)
- Mixed terminology (official military unit names, local nicknames, English transliterations)
- Incomplete information ("a military unit attacked" without identifying which one)

ENTITY EXTRACTION:
MILITARY UNITS:
- Patterns: "လိုင်း + number", "LID + number", "Infantry Battalion + number"
- Examples: "လိုင်း ၉၉" (LID 99), "ခရိုင် ၂၃၄" (Battalion 234)
- Cross-reference: rise/contexts/myanmar_tatmadaw_2024.db
- If match found → link to database entity, boost confidence
- If no match → flag as "potential_new_unit", requires verification

LOCATIONS:
- Villages, townships, regions (hierarchical: village → township → region)
- Database: rise/contexts/myanmar_townships.db
- Handle multiple spellings (Burmese transliteration varies: "Kale" vs "Kalay")
- Geocode when possible (lat/lon), note precision (village-level vs township-level)

PERSONS:
- Format: "Ko Aung" (civilian), "Captain Zaw" (military rank + name)
- WARNING: Burmese names are NOT first name + last name (different cultural system)
- Don't split "Ko Aung" into "first: Ko, last: Aung" - "Ko" is honorific
- Extract full name as written, note role (civilian, soldier, official)

EVENT EXTRACTION:
ATTACKS:
- Type: artillery_attack, small_arms_fire, aerial_bombardment, raid
- Required: date, location, perpetrator
- Optional: casualties, damage, weapons used
- Confidence factors: Specific details (time, weapon type) increase confidence

TEMPORAL EXTRACTION:
Myanmar calendar → Gregorian conversion:
- "၁၃၈၂ ခုနှစ် ကဆုန်လ ၁၅ ရက်" = "Year 1382, Month Kason, Day 15"
- Use myanmar-calendar library for conversion
- Result: 2021-04-15 (Gregorian)
- STORE BOTH versions (original Myanmar + converted Gregorian)
- Note precision (exact date vs "sometime in April")

CONFIDENCE SCORING RULES:
HIGH (0.85-1.0):
- Clear text, specific details, database match, multiple source corroboration
- Example: "လိုင်း ၉၉" mentioned, matches database, clear OCR, specific date

MEDIUM (0.65-0.84):
- Unclear text OR no database match OR missing details
- Example: "a military unit" (no specific ID) OR "LID XYZ" (not in database)

LOW (0.5-0.64):
- Very unclear OR single vague mention OR contradicted by other source
- Example: Blurry text, OR rumor ("I heard that..."), OR conflicts with other report

VERY LOW (< 0.5):
- Extract but FLAG FOR REVIEW
- Include in output but mark as requiring human verification

NEVER INVENT:
- Don't guess military unit numbers ("လိုင်း" without number is NOT "LID 99")
- Don't infer dates if not stated ("this happened" ≠ assume evidence date)
- Don't create casualty counts if not mentioned (absence of data ≠ zero casualties)

PROVENANCE REQUIREMENTS:
Every extraction includes:
{
  "entity": "Light Infantry Division 99",
  "source_evidence": "evd_042",
  "text_span": "လိုင်း ၉၉ အမှတ် ၁၄:၀၀ နာရီတွင် ရွာကို ပစ်ခတ်ခဲ့သည်",
  "character_offsets": [0, 8],  // Exactly where in source text
  "extraction_method": "ml_ner_burmese_v2.0 + context_database_match",
  "confidence": 0.95,
  "confidence_breakdown": {
    "ml_score": 0.88,
    "context_match_boost": +0.10,
    "ocr_quality_penalty": -0.03
  },
  "database_match": "tatmadaw_lid_99",
  "alternative_interpretations": []
}

SUCCESS: Legal investigator can trace extraction to exact source location, understand confidence reasoning, verify against original.
FAILURE: Hallucinated entities, hidden uncertainty, lost provenance, unjustified confidence scores.
```

---

## Implementation Strategy

### Step 1: Define the Mission
What is this tool supposed to accomplish in the forensic pipeline?

Not "extract entities" but "identify military units for legal accountability from messy Viber screenshots with uncertain encoding."

### Step 2: Describe the Reality
What does the actual data look like? Not ideal schemas, but real Viber screenshots with Zawgyi encoding and blurry photos.

### Step 3: Embed Domain Knowledge
What specific patterns, rules, databases, cultural context matter?

Myanmar military unit naming conventions, township hierarchies, calendar systems.

### Step 4: Establish Decision Criteria
How should the tool make judgment calls?

Explicit rules: "If X, then Y. If confidence < threshold, flag for review."

### Step 5: Define Success
What does good output look like? How will it be used?

"Legal investigator can submit to court" vs "analyst gets rough intelligence."

### Step 6: Write the Prompt
Combine all elements into a coherent mission briefing.

Test with sample data, refine based on output quality.

---

## Benefits Over Traditional Configuration

| Traditional Config | Brilliant Prompt |
|-------------------|------------------|
| `entity_types: [military_units]` | "Military units like 'LID 99', 'Battalion 234' - preserve Burmese terms, cross-reference database" |
| `confidence_threshold: 0.7` | "Score confidence 0.0-1.0 based on text clarity, database match, source corroboration. Legal accountability requires audit trail." |
| `language: burmese` | "Myanmar script (U+1000-109F), but beware Zawgyi encoding (broken font). Detect, convert, preserve original." |
| `output: json` | "JSON entity objects with complete provenance: source ID, character offsets, extraction method, confidence breakdown, alternatives." |

**Traditional config**: What to do
**Brilliant prompt**: Why, how, when, what if, what good looks like

---

## Adapting to New Contexts

To adapt RISE to non-Myanmar contexts, **rewrite the brilliant prompts** with new domain knowledge:

**Example: Syrian conflict documentation**

Change:
- Zawgyi/Unicode → Arabic script with various diacritics and regional spellings
- Myanmar calendar → Islamic calendar / Gregorian conversion
- Tatmadaw military units → Syrian Army / rebel group patterns
- Township databases → Syrian governorate/district hierarchies

Keep:
- Forensic integrity principles
- Confidence scoring approach
- Provenance requirements
- Uncertainty handling protocols

The **methodology** (phase-based processing, multi-dimensional output) is universal.
The **prompts** (domain knowledge, entity patterns) are context-specific.

---

## When NOT to Use Brilliant Prompts

Brilliant prompts work for:
- ✅ Complex domain knowledge (military unit recognition)
- ✅ Uncertain/messy data (Viber screenshots)
- ✅ High-stakes decisions (legal evidence)
- ✅ Adaptable AI models (LLMs that understand mission context)

Brilliant prompts are overkill for:
- ❌ Simple deterministic tasks (hash computation)
- ❌ Well-defined schemas (CSV with exact column specs)
- ❌ Low-stakes processing (analytics dashboards)
- ❌ Rule-based systems (regex patterns)

Use traditional config for deterministic tasks.
Use brilliant prompts for forensic judgment under uncertainty.

---

*"The prompt is not configuration. It's the encoding of forensic expertise into operational context."*
