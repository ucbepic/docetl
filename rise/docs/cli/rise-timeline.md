# rise-timeline: Temporal Analysis and Timeline Construction

## Mission

Transform verified events into chronological master timelines with temporal gap identification, date normalization (Myanmar calendar → Gregorian), and narrative sequencing for legal/investigative use.

## Core Philosophy

You are a **forensic chronologist**. Your job is to answer "What happened when?" with precision, uncertainty, and completeness. You:
- Order events strictly chronologically
- Normalize all dates to Gregorian (preserve originals)
- Identify temporal gaps (missing periods)
- Handle fuzzy dates (ranges, "sometime in April")
- Detect impossible sequences (temporal inconsistencies)
- Link related events (same incident, multi-day operations)
- Build navigable timelines (previous/next event chains)

**Rule: Explicit temporal precision always. "April 2021" ≠ "April 15, 2021" ≠ "April 15, 2021 14:00"**

---

## Usage

### Basic Timeline Construction
```bash
rise-timeline \
  --verified-events verified/events/ \
  --output timeline.json \
  --sort-by date \
  --format json
```

### Geographic-Filtered Timeline
```bash
rise-timeline \
  --verified-events verified/events/ \
  --location "Sagaing_Region_Kale_Township" \
  --date-range "2021-02-01:2024-12-31" \
  --output kale_timeline.json
```

### Multi-Resolution Timelines
```bash
rise-timeline \
  --verified-events verified/events/ \
  --resolution daily \
  --output timeline_daily.json

rise-timeline \
  --verified-events verified/events/ \
  --resolution monthly \
  --output timeline_monthly.json
```

### Timeline with Gap Analysis
```bash
rise-timeline \
  --verified-events verified/events/ \
  --identify-gaps \
  --gap-threshold 7  # Flag gaps > 7 days
  --output timeline_with_gaps.json
```

---

## Parameters

### Input Parameters

**`--verified-events <path>`** (required)
- Directory of verified events (from rise-verify)
- Can also be single JSON file containing event array

**`--date-range <start:end>`** (optional)
- Filter events to date range
- Format: `YYYY-MM-DD:YYYY-MM-DD`
- Example: `2021-02-01:2024-12-31`

**`--location <filter>`** (optional)
- Filter to specific geographic area
- Format: `Region`, `Region_Township`, `Region_Township_Village`
- Example: `Sagaing_Region_Kale_Township`

**`--event-types <types>`** (optional)
- Filter to specific event types
- Comma-separated: `attacks`, `arrests`, `displacements`
- Default: all types

### Timeline Construction

**`--resolution <level>`** (optional, default: event)
- `event`: One entry per event (most detailed)
- `daily`: Aggregate events by day
- `weekly`: Aggregate by week
- `monthly`: Aggregate by month
- `yearly`: Aggregate by year

**`--sort-by <field>`** (optional, default: date)
- `date`: Chronological order (primary sort)
- `confidence`: High-confidence events first
- `location`: Geographic clustering

**`--group-related`** (optional)
- Link events that are part of same incident/operation
- Example: Multi-day siege = one operation, multiple events

**`--narrative-mode`** (optional)
- Add narrative context between events
- "X days later...", "Meanwhile in Township Y...", "This began a pattern of..."

### Temporal Analysis

**`--identify-gaps`** (optional)
- Detect temporal gaps (periods with no events)
- Output gap analysis alongside timeline

**`--gap-threshold <days>`** (optional, default: 7)
- Minimum gap duration (in days) to flag
- Shorter gaps ignored (may be quiet periods)

**`--temporal-consistency-check`** (optional)
- Validate event sequences are logically possible
- Flag: "Event A 14:00 in Kale, Event B 14:30 in Yinmabin (75km away)" - impossible travel

**`--myanmar-calendar-display`** (optional)
- Show Myanmar calendar dates alongside Gregorian
- Format: "2021-04-15 (1382 Kason 15)"

### Output Parameters

**`--output <path>`** (required)
- Timeline output file
- Format: `.json`, `.csv`, `.html`, `.pdf`

**`--format <type>`** (optional, default: json)
- `json`: Structured timeline (for downstream processing)
- `csv`: Tabular format (for spreadsheets)
- `html`: Interactive web timeline
- `pdf`: Printable chronology report

**`--include-sources`** (optional)
- Embed source citations in timeline entries
- Useful for legal submissions

**`--confidence-threshold <float>`** (optional, default: 0.0)
- Minimum confidence to include event in timeline
- Example: `0.7` excludes low-confidence events

---

## Output Structure

### Timeline JSON Format

```json
{
  "timeline_id": "myanmar_sagaing_2021_2024",
  "timeline_version": "1.0",
  "creation_timestamp": "2024-01-11T10:00:00Z",
  "creation_tool": "rise-timeline v1.2.0",

  "metadata": {
    "geographic_scope": "Sagaing_Region",
    "temporal_scope": {
      "start_date": "2021-02-01",
      "end_date": "2024-12-31",
      "total_days": 1429
    },
    "event_statistics": {
      "total_events": 1247,
      "by_type": {
        "artillery_attack": 347,
        "village_raid": 289,
        "arrest": 156,
        "displacement": 234,
        "other": 221
      },
      "by_confidence": {
        "high": 892,
        "medium": 267,
        "low": 88
      }
    },
    "coverage_analysis": {
      "days_with_events": 412,
      "days_without_events": 1017,
      "coverage_percentage": 28.8,
      "longest_gap_days": 45
    }
  },

  "events": [
    {
      "timeline_entry_id": "tml_001",
      "event_id": "evt_attack_001",
      "sequence_number": 1,

      "temporal": {
        "date": "2021-02-01",
        "time": null,
        "timezone": "Asia/Yangon",
        "date_precision": "exact",
        "time_precision": null,
        "myanmar_calendar": "1382_Pyatho_17",
        "fuzzy_description": null,
        "original_text": "၂၀၂၁ ခုနှစ် ဖေဖော်ဝါရီလ ၁ ရက်"
      },

      "event_summary": {
        "type": "military_coup",
        "description": "Military coup announced, NLD government overthrown",
        "significance": "Initiating event of conflict period",
        "entities_involved": [
          {"type": "organization", "name": "Tatmadaw", "role": "perpetrator"},
          {"type": "person", "name": "Min Aung Hlaing", "role": "coup_leader"}
        ],
        "locations": [
          {"name": "Naypyidaw", "type": "capital_city"}
        ]
      },

      "confidence": {
        "overall": 1.0,
        "date_confidence": 1.0,
        "event_confidence": 1.0,
        "sources_count": 100
      },

      "sources": [
        {"evidence_id": "evd_001", "type": "official_announcement"},
        {"evidence_id": "evd_002", "type": "news_media"},
        {"evidence_id": "evd_003", "type": "witness_statement"}
      ],

      "related_events": {
        "previous": null,
        "next": "tml_002",
        "part_of_operation": null,
        "triggered_by": null,
        "led_to": ["tml_002", "tml_003", "tml_004"]
      },

      "tags": ["coup", "political", "national_level"]
    },

    {
      "timeline_entry_id": "tml_002",
      "event_id": "evt_attack_002",
      "sequence_number": 2,

      "temporal": {
        "date": "2021-02-06",
        "time": null,
        "timezone": "Asia/Yangon",
        "date_precision": "exact",
        "time_precision": null,
        "myanmar_calendar": "1382_Pyatho_22"
      },

      "event_summary": {
        "type": "protest",
        "description": "First major anti-coup protests in Yangon",
        "significance": "Beginning of Civil Disobedience Movement",
        "participants_estimated": 10000
      },

      "confidence": {
        "overall": 0.95,
        "date_confidence": 1.0,
        "event_confidence": 0.95,
        "sources_count": 25
      },

      "sources": [
        {"evidence_id": "evd_045", "type": "photo"},
        {"evidence_id": "evd_046", "type": "witness_statement"}
      ],

      "related_events": {
        "previous": "tml_001",
        "next": "tml_003",
        "triggered_by": "tml_001"
      },

      "tags": ["protest", "cdm", "yangon"]
    },

    {
      "timeline_entry_id": "tml_234",
      "event_id": "evt_attack_234",
      "sequence_number": 234,

      "temporal": {
        "date": "2021-04-15",
        "time": "14:00:00",
        "timezone": "Asia/Yangon",
        "date_precision": "exact",
        "time_precision": "approximate_hour",
        "myanmar_calendar": "1382_Kason_15"
      },

      "event_summary": {
        "type": "artillery_attack",
        "description": "LID 99 shelled Thayetchaung Village, Kale Township",
        "entities_involved": [
          {"type": "military_unit", "name": "LID 99", "role": "perpetrator"},
          {"type": "person", "name": "Ko Aung", "role": "injured_civilian"}
        ],
        "locations": [
          {"name": "Thayetchaung_Village", "township": "Kale", "region": "Sagaing"}
        ],
        "casualties": {
          "killed": 0,
          "injured": 1,
          "confidence": 0.8
        }
      },

      "confidence": {
        "overall": 0.88,
        "date_confidence": 0.95,
        "time_confidence": 0.75,
        "event_confidence": 0.90,
        "sources_count": 3
      },

      "sources": [
        {"evidence_id": "evd_042", "type": "viber_screenshot"},
        {"evidence_id": "evd_118", "type": "witness_statement"},
        {"evidence_id": "evd_119", "type": "follow_up_report"}
      ],

      "related_events": {
        "previous": "tml_233",
        "next": "tml_235",
        "part_of_operation": "sagaing_clearance_ops_april_2021"
      },

      "tags": ["attack", "artillery", "sagaing", "lid_99", "civilian_casualties"]
    }
  ],

  "temporal_gaps": [
    {
      "gap_id": "gap_001",
      "gap_type": "missing_documentation",
      "start_date": "2021-03-15",
      "end_date": "2021-03-22",
      "duration_days": 7,
      "location": "Kale_Township",
      "context": "Known military operations in area but no documentation",
      "events_before": "tml_078",
      "events_after": "tml_089",
      "priority": "high",
      "collection_recommendations": [
        "Contact CDM networks in Kale for this period",
        "Review satellite imagery 2021-03-15 to 2021-03-22",
        "Interview IDPs who fled during this time"
      ]
    },

    {
      "gap_id": "gap_002",
      "gap_type": "quiet_period",
      "start_date": "2021-06-01",
      "end_date": "2021-06-10",
      "duration_days": 9,
      "location": "Yinmabin_Township",
      "context": "No events documented - may be actual quiet period or missing data",
      "priority": "medium"
    }
  ],

  "temporal_inconsistencies": [
    {
      "inconsistency_id": "tinc_001",
      "type": "impossible_travel_time",
      "description": "Same unit reported in two locations without sufficient travel time",
      "event_a": {
        "timeline_entry_id": "tml_456",
        "date": "2021-05-10",
        "time": "14:00",
        "location": "Kale_Township",
        "entity": "LID_99"
      },
      "event_b": {
        "timeline_entry_id": "tml_457",
        "date": "2021-05-10",
        "time": "14:30",
        "location": "Yinmabin_Township",
        "entity": "LID_99"
      },
      "analysis": {
        "distance_km": 75,
        "time_available_minutes": 30,
        "required_travel_time_minutes": 120,
        "conclusion": "Impossible - likely different sub-units or date error in one report"
      },
      "resolution_status": "requires_human_review"
    }
  ],

  "narrative_summary": {
    "period": "2021-02-01 to 2024-12-31",
    "key_phases": [
      {
        "phase_name": "Coup and Initial Response",
        "date_range": "2021-02-01 to 2021-03-31",
        "events_count": 89,
        "description": "Military coup, protests emerge, early armed resistance forms"
      },
      {
        "phase_name": "Sagaing Escalation",
        "date_range": "2021-04-01 to 2021-12-31",
        "events_count": 456,
        "description": "Heavy military operations in Sagaing, village burnings, mass displacement"
      }
    ]
  }
}
```

---

## Timeline Construction Process

### Phase 1: Event Ingestion
```
Verified Events (JSON) →
  Load all events
  Validate date formats
  Parse Myanmar calendar dates
  Convert to Gregorian (preserve originals)
```

### Phase 2: Temporal Normalization
```
For each event:
  Extract date (required)
  Extract time (if available)
  Determine precision (exact vs approximate vs range)
  Normalize timezone (→ Asia/Yangon)
  Convert Myanmar calendar → Gregorian
  Store both versions
```

### Phase 3: Sorting & Sequencing
```
Sort events:
  Primary: date (ascending)
  Secondary: time (if available)
  Tertiary: confidence (high first if same date/time)

Assign sequence numbers
Link previous/next events
```

### Phase 4: Gap Identification
```
For each consecutive event pair:
  time_gap = event[n+1].date - event[n].date

  if time_gap > gap_threshold:
    Check if same location
    Check if known quiet period OR missing documentation
    Categorize gap (missing_docs vs quiet_period vs different_area)
    Assess priority for collection
```

### Phase 5: Consistency Validation
```
For events with same entity:
  Check temporal sequence makes sense
  Validate travel times between locations
  Flag impossible sequences

For events with same location:
  Check for contradictory simultaneous events
```

### Phase 6: Relationship Mapping
```
Identify related events:
  - Multi-day operations (siege, clearance ops)
  - Cause-effect chains (attack → displacement → IDP camp)
  - Simultaneous events (coordinated attacks)

Build event graphs
```

---

## Date Normalization

### Myanmar Calendar Conversion

**Myanmar Calendar System**:
- Lunar calendar (12-13 months per year)
- Month names: Tagu, Kason, Nayon, Waso, Wagaung, Tawthalin, Thadingyut, Tazaungmon, Nadaw, Pyatho, Tabodwe, Tabaung
- Year count: Myanmar Era (ME) = Gregorian - 638
- Intercalary months (13th month in some years)

**Conversion Process**:
```python
from myanmar_calendar import MyanmarDate

# Input: "၁၃၈၂ ခုနှစ် ကဆုန်လ ၁၅ ရက်"
myanmar_date = MyanmarDate(
    year=1382,  # Myanmar Era
    month="Kason",
    day=15
)

gregorian_date = myanmar_date.to_gregorian()
# Result: 2021-04-15

# Store both
timeline_entry = {
    "date": "2021-04-15",
    "myanmar_calendar": "1382_Kason_15",
    "conversion_confidence": 0.98
}
```

### Fuzzy Date Handling

**Examples**:
```json
{
  "original_text": "sometime in April 2021",
  "date": "2021-04-15",  // Mid-month estimate
  "date_precision": "month",
  "date_range": {"start": "2021-04-01", "end": "2021-04-30"}
},

{
  "original_text": "a few days after the village raid",
  "date": "2021-04-18",  // Estimated based on raid date + "few days"
  "date_precision": "estimated_relative",
  "date_range": {"start": "2021-04-17", "end": "2021-04-20"},
  "reference_event": "evt_raid_045"
}
```

---

## Gap Analysis

### Gap Types

**1. Missing Documentation**
- Known events occurred (military operations visible in satellite imagery)
- But no ground-level documentation
- High priority for collection

**2. Quiet Periods**
- No events documented
- May be actual lull in violence
- Lower priority (still worth verifying)

**3. Geographic Gaps**
- Events documented in Township A
- Nothing from Township B (adjacent)
- Suspicious - likely missing, not actually quiet

### Gap Prioritization

```python
def prioritize_gap(gap):
    score = 0

    # Duration
    if gap.duration_days > 30:
        score += 3
    elif gap.duration_days > 14:
        score += 2
    elif gap.duration_days > 7:
        score += 1

    # Context
    if gap.context == "known_operations_but_no_docs":
        score += 3
    elif gap.context == "adjacent_areas_have_events":
        score += 2

    # Location importance
    if gap.location in high_conflict_areas:
        score += 2

    # Priority
    if score >= 6:
        return "critical"
    elif score >= 4:
        return "high"
    elif score >= 2:
        return "medium"
    else:
        return "low"
```

---

## Integration with Other Tools

### From Verification to Timeline
```bash
rise-verify --entities extracted/ --events extracted/ --output verified/
rise-timeline --verified-events verified/events/ --output timeline.json
```

### Timeline to Dossier Cross-Reference
```bash
rise-timeline --verified-events verified/ --output timeline.json
rise-dossier --theme military_operations --cross-ref timeline.json --output dossiers/
```

---

## The Brilliant Prompt

> **You are a forensic chronologist constructing a master timeline from Myanmar conflict evidence.**
>
> Mission: Order events chronologically with EXPLICIT PRECISION about dates, identify temporal gaps, detect impossible sequences.
>
> Critical tasks:
> 1. **Normalize dates** - Myanmar calendar → Gregorian (use myanmar-calendar library). Store both versions.
> 2. **Preserve precision** - "April 2021" ≠ "April 15, 2021" ≠ "April 15, 2021 14:00". Never pretend more precision than source provides.
> 3. **Handle fuzzy dates** - "a few days later" becomes date range, not exact date. Document estimation method.
> 4. **Link events** - Multi-day siege? Link as one operation. Attack → displacement → IDP camp? Show cause-effect chain.
> 5. **Identify gaps** - No events for 2 weeks in high-conflict area? Flag as "missing_documentation", not "quiet_period".
> 6. **Validate consistency** - Unit X in Town A at 14:00, Town B (50km away) at 14:30? IMPOSSIBLE - flag for review.
> 7. **Sequence strictly** - Chronological order is sacred. Never re-order for narrative convenience.
>
> Output: Timeline JSON with sequence numbers, previous/next links, temporal gaps, consistency checks, Myanmar calendar preserved alongside Gregorian.
>
> Success: Legal investigator can construct precise chronology for court, analyst can identify collection gaps, researcher can understand event flow.

---

*rise-timeline is the chronological backbone of RISE. Every other output dimension cross-references back to this temporal structure.*
