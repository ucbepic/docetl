# rise-dossier: Thematic Intelligence Compilation

## Mission

Organize verified events and entities into thematic dossiers - topic-focused intelligence products that answer "Tell me everything about X" (military operations, human rights violations, displacement patterns, etc.).

## Core Philosophy

You are a **thematic intelligence compiler**. Different users need different organizational structures:
- Human rights researchers need violations dossier (all torture, all arbitrary detention)
- Military analysts need operations dossier (all LID 99 activities, all artillery usage)
- Displacement trackers need migration dossier (all forced relocations, IDP camp populations)

**Same events, multiple dossiers.** One artillery attack appears in: Military Operations dossier, Human Rights Violations dossier, Displacement dossier (if it caused displacement).

**Rule: Cross-reference everything. User jumps from dossier → timeline → archive → back.**

---

## Usage

### Single Thematic Dossier
```bash
rise-dossier \
  --theme military_operations \
  --verified-events verified/events/ \
  --verified-entities verified/entities/ \
  --output dossiers/military_operations.json
```

### Multiple Dossiers at Once
```bash
rise-dossier \
  --themes military_operations,human_rights_violations,displacement,economic_impact \
  --verified-events verified/ \
  --verified-entities verified/ \
  --output dossiers/
```

### Geographic-Focused Dossier
```bash
rise-dossier \
  --theme military_operations \
  --location "Sagaing_Region_Kale_Township" \
  --date-range "2021-02-01:2024-12-31" \
  --output dossiers/kale_military_ops.pdf
```

### Actor-Focused Dossier
```bash
rise-dossier \
  --theme military_operations \
  --actor "LID_99" \
  --output dossiers/lid99_activities.json
```

---

## Parameters

### Input Parameters

**`--verified-events <path>`** (required)
- Directory of verified events (from rise-verify)

**`--verified-entities <path>`** (required)
- Directory of verified entities

**`--timeline <path>`** (optional)
- Timeline JSON for cross-referencing
- Enables "show this entry in timeline context"

### Thematic Focus

**`--theme <name>`** or **`--themes <list>`** (required)
- Single theme or comma-separated list
- Standard themes:
  - `military_operations`
  - `human_rights_violations`
  - `displacement`
  - `economic_impact`
  - `resistance_activities`
  - `political_developments`
  - `humanitarian_response`
  - `media_information`
- Custom themes via theme definition file

**`--theme-definition <path>`** (optional)
- Custom theme rules (what events/entities belong to this theme)
- JSON file defining inclusion criteria

### Filtering

**`--location <filter>`** (optional)
- Geographic scope
- Example: `Sagaing_Region` or `Kale_Township`

**`--actor <entity>`** (optional)
- Focus on specific actor (military unit, organization, person)
- Example: `LID_99` or `NUG` or `PDF_Kale`

**`--date-range <start:end>`** (optional)
- Temporal scope
- Format: `YYYY-MM-DD:YYYY-MM-DD`

**`--confidence-threshold <float>`** (optional, default: 0.5)
- Minimum confidence for inclusion

### Output Parameters

**`--output <path>`** (required)
- Dossier output path
- Format: `.json`, `.pdf`, `.html`, `.docx`

**`--format <type>`** (optional, default: json)
- `json`: Structured data (for downstream processing)
- `pdf`: Professional report (for distribution)
- `html`: Interactive dossier (for web)
- `docx`: Editable document (for human editing)

**`--include-sources`** (optional)
- Embed source citations
- Essential for legal/research use

**`--summary-statistics`** (optional)
- Include aggregate statistics (total incidents, casualties, locations affected)

---

## Output Structure

### Dossier JSON Format

```json
{
  "dossier_id": "dossier_military_ops_sagaing_2021_2024",
  "dossier_version": "1.0",
  "creation_timestamp": "2024-01-11T11:00:00Z",
  "creation_tool": "rise-dossier v1.2.0",

  "metadata": {
    "theme": "military_operations",
    "subthemes": ["artillery_attacks", "village_raids", "aerial_bombardment", "forced_labor"],
    "geographic_scope": "Sagaing_Region",
    "temporal_scope": {
      "start_date": "2021-02-01",
      "end_date": "2024-12-31"
    },
    "actor_focus": null,
    "confidence_threshold": 0.5
  },

  "summary_statistics": {
    "total_incidents": 892,
    "incident_breakdown": {
      "artillery_attacks": 347,
      "village_raids": 289,
      "aerial_bombardment": 134,
      "forced_labor": 122
    },
    "temporal_distribution": {
      "2021": 234,
      "2022": 356,
      "2023": 245,
      "2024": 57
    },
    "geographic_distribution": {
      "Kale_Township": 234,
      "Yinmabin_Township": 189,
      "Tamu_Township": 145
    },
    "actors_involved": {
      "perpetrators": [
        {"entity": "LID_99", "incidents": 234},
        {"entity": "LID_88", "incidents": 189},
        {"entity": "LID_77", "incidents": 156}
      ]
    },
    "casualties": {
      "killed": 345,
      "injured": 892,
      "missing": 123,
      "confidence_note": "Based on available evidence; actual numbers likely higher"
    }
  },

  "entries": [
    {
      "entry_id": "dossier_entry_001",
      "incident_type": "artillery_attack",
      "entry_number": 1,

      "temporal": {
        "date": "2021-04-15",
        "time": "14:00",
        "myanmar_calendar": "1382_Kason_15"
      },

      "location": {
        "primary": "Thayetchaung_Village_Kale_Township_Sagaing_Region",
        "coordinates": {"lat": 23.2, "lon": 94.1, "precision": "village"},
        "affected_area_km": 2
      },

      "incident_description": {
        "summary": "LID 99 Artillery Unit shelled Thayetchaung Village with estimated 15-20 rounds",
        "detailed_narrative": "At approximately 14:00 on April 15, 2021, Light Infantry Division 99 Artillery Unit conducted a shelling operation targeting Thayetchaung Village in Kale Township. Witnesses reported hearing approximately 15-20 artillery rounds impacting the village over a 30-minute period. The attack damaged residential buildings and a local monastery. One civilian, Ko Aung, was injured by shrapnel.",

        "perpetrators": [
          {
            "entity_id": "ent_mil_001",
            "name": "Light Infantry Division 99 Artillery Unit",
            "role": "attacking_force",
            "confidence": 0.95
          }
        ],

        "victims": [
          {
            "entity_id": "ent_per_001",
            "name": "Ko Aung",
            "role": "injured_civilian",
            "status": "injured",
            "confidence": 0.9
          }
        ],

        "witnesses": [
          {
            "entity_id": "ent_per_002",
            "role": "eyewitness",
            "confidence": 0.85
          }
        ],

        "tactics_weapons": {
          "weapon_type": "artillery",
          "estimated_rounds": "15-20",
          "tactics": ["indirect_fire", "civilian_area_targeting"]
        },

        "impact": {
          "casualties": {"killed": 0, "injured": 1},
          "damage": [
            {"type": "residential_building", "count": 3, "severity": "destroyed"},
            {"type": "monastery", "count": 1, "severity": "damaged"}
          ],
          "displacement": {"families": 12, "individuals": 45},
          "economic_loss_usd": "estimated_15000"
        }
      },

      "evidence_base": {
        "primary_sources": [
          {"evidence_id": "evd_042", "type": "viber_screenshot", "confidence": 0.95},
          {"evidence_id": "evd_118", "type": "witness_statement", "confidence": 0.92},
          {"evidence_id": "evd_119", "type": "follow_up_report", "confidence": 0.88}
        ],
        "corroboration_level": "multi_source",
        "verification_status": "corroborated",
        "confidence_overall": 0.88
      },

      "cross_references": {
        "timeline_entry": "tml_234",
        "related_dossiers": ["human_rights_violations", "displacement"],
        "related_entries": ["dossier_entry_045", "dossier_entry_067"],
        "primary_archive": "archive_evd_042"
      },

      "legal_relevance": {
        "potential_violations": [
          "Indiscriminate attack (IHL violation)",
          "Attack on protected objects (monastery - cultural property)"
        ],
        "documentation_quality": "sufficient_for_preliminary_investigation",
        "additional_evidence_needed": [
          "Satellite imagery confirmation",
          "Medical records for injured civilian",
          "Monastery damage assessment"
        ]
      },

      "tags": ["artillery", "civilian_casualties", "property_destruction", "lid_99", "kale_township"]
    }
  ],

  "patterns_analysis": {
    "temporal_patterns": [
      {
        "pattern": "Artillery attacks increased significantly in April-May 2021",
        "evidence": "87 attacks in Apr-May vs 34 in Feb-Mar",
        "significance": "Indicates escalation following coup resistance"
      }
    ],

    "geographic_patterns": [
      {
        "pattern": "Kale Township disproportionately targeted",
        "evidence": "234 incidents in Kale vs 89 average for other townships",
        "significance": "Potential systematic targeting of PDF stronghold"
      }
    ],

    "tactical_patterns": [
      {
        "pattern": "Artillery used primarily against civilian villages, rarely against PDF positions",
        "evidence": "78% of artillery attacks on villages, 22% on known PDF locations",
        "significance": "Questions military necessity, suggests civilian intimidation tactics"
      }
    ],

    "actor_patterns": [
      {
        "pattern": "LID 99 responsible for majority of Kale Township attacks",
        "evidence": "234 of 289 Kale incidents attributed to LID 99",
        "significance": "Identify command responsibility for accountability"
      }
    ]
  },

  "gaps_identified": [
    {
      "gap_type": "missing_actor_detail",
      "description": "LID 99 commander name unknown for 2021-2022 period",
      "priority": "high",
      "relevance": "Command responsibility determination"
    },
    {
      "gap_type": "casualty_undercount",
      "description": "Many incidents report '3+ injured' without exact counts",
      "priority": "medium",
      "relevance": "Total casualty statistics likely underestimate actual harm"
    }
  ],

  "methodology_notes": {
    "inclusion_criteria": "All events classified as military_operations theme",
    "exclusion_criteria": "Events with confidence < 0.5",
    "confidence_weighting": "Higher confidence events featured more prominently",
    "source_diversity": "892 incidents from 342 unique sources"
  }
}
```

---

## Standard Themes

### 1. Military Operations
**Includes**: Attacks, raids, sieges, troop movements, supply operations
**Subthemes**: Artillery attacks, aerial bombardment, ground raids, naval operations
**Key metrics**: Incidents count, units involved, geographic spread, casualties

### 2. Human Rights Violations
**Includes**: Torture, arbitrary detention, extrajudicial killings, forced labor, sexual violence
**Subthemes**: By violation type (torture, detention, etc.)
**Key metrics**: Victim count, perpetrator patterns, violation types, legal classification

### 3. Displacement
**Includes**: Forced relocations, IDP camps, refugee flows, return patterns
**Subthemes**: Causes of displacement, IDP demographics, camp conditions
**Key metrics**: Displaced populations, locations affected, duration, humanitarian needs

### 4. Economic Impact
**Includes**: Looting, business destruction, crop burning, infrastructure damage, trade disruption
**Subthemes**: By economic sector (agriculture, commerce, infrastructure)
**Key metrics**: Economic loss estimates, livelihoods affected, recovery timeline

### 5. Resistance Activities
**Includes**: PDF operations, urban resistance, civil disobedience
**Subthemes**: By resistance type and actor
**Key metrics**: Operation count, effectiveness, casualties, geographic scope

### 6. Political Developments
**Includes**: CRPH/NUG activities, local governance, political arrests
**Subthemes**: Governance structures, policy announcements, political prisoners
**Key metrics**: Political milestones, governance reach, political prisoner counts

### 7. Humanitarian Response
**Includes**: Aid delivery, medical services, protection activities, shelter provision
**Subthemes**: By aid type and provider
**Key metrics**: Beneficiaries reached, aid volume, coverage gaps

### 8. Media & Information
**Includes**: Propaganda, censorship, documentation efforts, disinformation
**Subthemes**: By information type and source
**Key metrics**: Media restrictions, propaganda patterns, documentation coverage

---

## Cross-Dossier Entries

**Same event, multiple dossiers**:

```json
{
  "event": "LID 99 artillery attack on Thayetchaung Village",
  "appears_in_dossiers": [
    {
      "dossier": "military_operations",
      "angle": "Military tactics and units involved",
      "entry_focus": "Artillery usage patterns by LID 99"
    },
    {
      "dossier": "human_rights_violations",
      "angle": "Indiscriminate attack on civilians",
      "entry_focus": "IHL violations and civilian harm"
    },
    {
      "dossier": "displacement",
      "angle": "Caused displacement of 12 families",
      "entry_focus": "Attack-driven displacement patterns"
    }
  ]
}
```

Each dossier emphasizes different aspects, but all cross-reference the same source evidence.

---

## Pattern Analysis

Dossiers identify patterns across incidents:

### Temporal Pattern Detection
```python
def detect_temporal_patterns(incidents):
    # Group by month
    monthly_counts = group_by_month(incidents)

    # Identify spikes
    mean = average(monthly_counts)
    spikes = [month for month, count in monthly_counts if count > mean * 1.5]

    # Analyze
    pattern = {
        "pattern": f"Spike in {', '.join(spikes)}",
        "evidence": f"Count: {monthly_counts[spikes[0]]} vs average {mean}",
        "significance": "Identify escalation periods"
    }
    return pattern
```

### Geographic Clustering
```python
def detect_geographic_patterns(incidents):
    # Group by township
    township_counts = group_by_township(incidents)

    # Identify hotspots
    hotspots = top_n(township_counts, n=3)

    # Analyze
    pattern = {
        "pattern": f"{hotspots[0]} disproportionately affected",
        "evidence": f"{township_counts[hotspots[0]]} incidents vs {mean(township_counts)} average",
        "significance": "Potential systematic targeting"
    }
    return pattern
```

### Tactical Pattern Recognition
```python
def detect_tactical_patterns(incidents):
    # Analyze weapon usage
    weapons = [inc.weapon_type for inc in incidents]
    primary_weapon = most_common(weapons)

    # Analyze target types
    targets = [inc.target_type for inc in incidents]
    civilian_percentage = count(targets, "civilian") / len(targets)

    # Analyze
    if civilian_percentage > 0.7:
        pattern = {
            "pattern": f"{primary_weapon} used primarily against civilians",
            "evidence": f"{civilian_percentage*100}% of attacks on civilian targets",
            "significance": "Questions military necessity, suggests intimidation"
        }
    return pattern
```

---

## Integration with Other Outputs

### Dossier ↔ Timeline
```bash
# Create timeline
rise-timeline --verified-events verified/ --output timeline.json

# Create dossier with timeline cross-refs
rise-dossier --theme military_operations --timeline timeline.json --output dossier.json

# Dossier entries include timeline_entry IDs
# Users can jump from thematic view → chronological context
```

### Dossier ↔ Archive
```bash
# Create archive
rise-archive --evidence bundles/ --output archive/

# Create dossier with archive cross-refs
rise-dossier --theme human_rights_violations --archive archive/ --output dossier.json

# Dossier entries link to primary source archives
# Users can jump from incident description → original evidence
```

---

## The Brilliant Prompt

> **You are a thematic intelligence compiler creating focused dossiers from Myanmar conflict evidence.**
>
> Mission: Organize events by theme to answer "Tell me everything about X" - military operations, violations, displacement, etc.
>
> Critical tasks:
> 1. **Theme classification** - Which events belong in this dossier? Use theme definition rules.
> 2. **Cross-dossier entries** - Same artillery attack appears in: Military Ops dossier (tactics focus), HR Violations dossier (IHL violations focus), Displacement dossier (caused flight). Different angles, same evidence.
> 3. **Pattern detection** - Analyze incidents collectively. Temporal spikes? Geographic clustering? Tactical patterns? Actor patterns?
> 4. **Summary statistics** - Total incidents, casualty counts, locations affected, perpetrators involved.
> 5. **Gap identification** - What's missing in this thematic area? Unknown commanders? Underreported violations? Coverage gaps?
> 6. **Cross-reference everything** - Every entry links to: timeline (chronological context), archive (original evidence), related dossiers (other perspectives).
>
> Output: Thematic dossier (JSON/PDF) with entries, patterns, statistics, gaps, full cross-references.
>
> Success: Researcher can understand complete picture of theme, identify patterns, jump to timeline/archive for details, know what evidence gaps exist.

---

*rise-dossier is the thematic organization layer. Same evidence base, multiple perspectives, all cross-referenced.*
