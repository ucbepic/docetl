# Myanmar Context Databases

Context databases are the **domain knowledge infrastructure** of RISE. They enable:
- Entity recognition (is "LID 99" a known military unit?)
- Validation (does "Thayetchaung Village" exist in Kale Township?)
- Confidence boosting (database match → higher confidence)
- Gap identification (unit mentioned but not in database → potential new unit)

For Myanmar conflict documentation, context databases cover military structures, geography, calendar systems, and terminology.

---

## Database Structure

### Directory Layout

```
rise/contexts/myanmar/
├── README.md
├── military/
│   ├── tatmadaw_units.json
│   ├── unit_aliases.json
│   ├── command_structure.json
│   └── known_commanders.json
├── geography/
│   ├── regions.json
│   ├── townships.json
│   ├── villages.json
│   └── geocoding.json
├── calendar/
│   ├── myanmar_calendar_mapping.json
│   └── holiday_dates.json
├── terminology/
│   ├── military_terms.json
│   ├── political_terms.json
│   └── zawgyi_unicode_pairs.json
└── metadata.json
```

---

## Military Context Database

### Tatmadaw Units (`military/tatmadaw_units.json`)

```json
{
  "database_id": "myanmar_tatmadaw_units_2024",
  "version": "2.1",
  "last_updated": "2024-01-01",
  "description": "Myanmar military (Tatmadaw) organizational units",
  "sources": [
    "Myanmar Peace Monitor",
    "Myanmar Military organizational documents",
    "Open-source intelligence compilation"
  ],

  "units": [
    {
      "unit_id": "tatmadaw_lid_99",
      "unit_type": "light_infantry_division",
      "official_name_english": "Light Infantry Division 99",
      "official_name_burmese": "ပေါ့ပါးခြေလျင်တပ်မတော် ၉၉",
      "common_names": [
        "LID 99",
        "99th Division",
        "Division 99",
        "လိုင်း ၉၉",
        "၉၉ တပ်မ"
      ],

      "organizational_hierarchy": {
        "parent_command": "Northern Command",
        "parent_command_id": "tatmadaw_northern_command",
        "subordinate_units": [
          {
            "unit_id": "tatmadaw_lid99_bn1",
            "name": "Infantry Battalion 1",
            "type": "infantry_battalion"
          },
          {
            "unit_id": "tatmadaw_lid99_bn2",
            "name": "Infantry Battalion 2",
            "type": "infantry_battalion"
          },
          {
            "unit_id": "tatmadaw_lid99_arty",
            "name": "Artillery Unit",
            "type": "artillery_unit"
          }
        ]
      },

      "known_commanders": [
        {
          "name": "Unknown",
          "period": "2021-2022",
          "confidence": "unknown",
          "notes": "Commander name not publicly available for this period"
        }
      ],

      "operational_areas": [
        "Sagaing_Region_Kale_Township",
        "Sagaing_Region_Tamu_Township",
        "Sagaing_Region_Yinmabin_Township"
      ],

      "active_period": {
        "established": "1990s",
        "active_status": "active",
        "last_confirmed_activity": "2024-01-01"
      },

      "notes": "LID 99 has been heavily deployed in Sagaing Region post-coup, particularly in Kale Township. Responsible for numerous documented attacks on civilian villages.",

      "verification_level": "confirmed",
      "last_verified": "2024-01-01"
    },

    {
      "unit_id": "tatmadaw_lid_88",
      "unit_type": "light_infantry_division",
      "official_name_english": "Light Infantry Division 88",
      "official_name_burmese": "ပေါ့ပါးခြေလျင်တပ်မတော် ၈၈",
      "common_names": [
        "LID 88",
        "88th Division",
        "လိုင်း ၈၈"
      ],

      "organizational_hierarchy": {
        "parent_command": "Northern Command",
        "subordinate_units": []
      },

      "operational_areas": [
        "Sagaing_Region_Kale_Township",
        "Sagaing_Region_Kalay_Township"
      ],

      "active_period": {
        "established": "1980s",
        "active_status": "active"
      },

      "verification_level": "confirmed",
      "last_verified": "2024-01-01"
    }
  ],

  "unit_types": [
    {
      "type_id": "light_infantry_division",
      "name_english": "Light Infantry Division",
      "name_burmese": "ပေါ့ပါးခြေလျင်တပ်မတော်",
      "abbreviation": "LID",
      "typical_size": "3000-5000 personnel",
      "typical_structure": ["Infantry Battalions", "Artillery Unit", "Support Units"]
    },
    {
      "type_id": "infantry_battalion",
      "name_english": "Infantry Battalion",
      "name_burmese": "ခြေလျင်တပ်ရင်း",
      "abbreviation": "IB",
      "typical_size": "500-800 personnel"
    }
  ]
}
```

### Unit Aliases (`military/unit_aliases.json`)

Maps common variations to canonical unit IDs:

```json
{
  "aliases": {
    "LID 99": "tatmadaw_lid_99",
    "99th Division": "tatmadaw_lid_99",
    "Division 99": "tatmadaw_lid_99",
    "လိုင်း ၉၉": "tatmadaw_lid_99",
    "၉၉ တပ်မ": "tatmadaw_lid_99",
    "Light Infantry Division 99": "tatmadaw_lid_99",

    "LID 88": "tatmadaw_lid_88",
    "88th Division": "tatmadaw_lid_88",
    "လိုင်း ၈၈": "tatmadaw_lid_88"
  },

  "fuzzy_patterns": [
    {
      "pattern": "LID\\s+(\\d+)",
      "unit_id_template": "tatmadaw_lid_{number}",
      "description": "Light Infantry Division number pattern"
    },
    {
      "pattern": "လိုင်း\\s+([၀-၉]+)",
      "unit_id_template": "tatmadaw_lid_{number}",
      "description": "LID in Burmese (Myanmar numerals)"
    }
  ]
}
```

---

## Geographic Context Database

### Townships (`geography/townships.json`)

```json
{
  "database_id": "myanmar_townships_2024",
  "version": "1.5",
  "last_updated": "2024-01-01",

  "regions": [
    {
      "region_id": "sagaing",
      "name_english": "Sagaing Region",
      "name_burmese": "စစ်ကိုင်းတိုင်းဒေသကြီး",
      "iso_code": "MMR-01",

      "townships": [
        {
          "township_id": "sagaing_kale",
          "name_english": "Kale Township",
          "name_burmese": "ကလေးမြို့နယ်",
          "alternative_spellings": ["Kalay", "Kale", "Kalemyo"],

          "coordinates": {
            "lat": 23.183,
            "lon": 94.083,
            "precision": "township_center"
          },

          "area_km2": 1821,
          "population_2014": 143500,

          "adjacent_townships": [
            "sagaing_tamu",
            "sagaing_yinmabin",
            "chin_tonzang"
          ],

          "conflict_status": {
            "intensity": "high",
            "last_updated": "2024-01-01",
            "notes": "Heavy military operations post-coup, significant PDF presence"
          },

          "villages_count": 156
        },

        {
          "township_id": "sagaing_yinmabin",
          "name_english": "Yinmabin Township",
          "name_burmese": "အင်းမ သင်မြို့နယ်",
          "alternative_spellings": ["Yinmarbin", "Yinmabin"],

          "coordinates": {
            "lat": 22.25,
            "lon": 94.75,
            "precision": "township_center"
          },

          "adjacent_townships": [
            "sagaing_kale",
            "sagaing_salingyi"
          ],

          "conflict_status": {
            "intensity": "high",
            "last_updated": "2024-01-01"
          }
        }
      ]
    }
  ]
}
```

### Villages (`geography/villages.json`)

```json
{
  "villages": [
    {
      "village_id": "sagaing_kale_thayetchaung",
      "township_id": "sagaing_kale",
      "name_english": "Thayetchaung Village",
      "name_burmese": "သရက်ချောင်းရွာ",
      "alternative_spellings": ["Tayetchaung", "Thayetchaung", "Thayet Chaung"],

      "coordinates": {
        "lat": 23.2,
        "lon": 94.1,
        "precision": "village",
        "source": "OpenStreetMap"
      },

      "village_tract": "Thayetchaung Village Tract",

      "population_estimate": {
        "count": 450,
        "year": 2020,
        "source": "estimated"
      },

      "conflict_events": {
        "documented_events": 12,
        "first_event_date": "2021-04-15",
        "last_event_date": "2023-11-20",
        "event_types": ["artillery_attack", "raid", "displacement"]
      },

      "status": {
        "displacement": "partial",
        "destruction": "significant_damage",
        "last_updated": "2024-01-01"
      }
    }
  ]
}
```

---

## Calendar Context Database

### Myanmar Calendar Mapping (`calendar/myanmar_calendar_mapping.json`)

```json
{
  "database_id": "myanmar_calendar_2024",
  "description": "Myanmar (Burmese) calendar to Gregorian conversion data",

  "calendar_system": {
    "type": "lunisolar",
    "epoch": "Myanmar Era (ME)",
    "me_to_gregorian_offset": 638,
    "note": "Myanmar year = Gregorian year - 638"
  },

  "months": [
    {
      "month_number": 1,
      "name_burmese": "တန်ခူး",
      "name_romanized": "Tagu",
      "gregorian_approximate": "March-April"
    },
    {
      "month_number": 2,
      "name_burmese": "ကဆုန်",
      "name_romanized": "Kason",
      "gregorian_approximate": "April-May"
    },
    {
      "month_number": 3,
      "name_burmese": "နယုန်",
      "name_romanized": "Nayon",
      "gregorian_approximate": "May-June"
    }
  ],

  "conversion_examples": [
    {
      "myanmar_date": "1382 ကဆုန် 15",
      "myanmar_year": 1382,
      "myanmar_month": "Kason",
      "myanmar_day": 15,
      "gregorian_date": "2021-04-15",
      "confidence": 1.0
    },
    {
      "myanmar_date": "1383 တန်ခူး 1",
      "myanmar_year": 1383,
      "myanmar_month": "Tagu",
      "myanmar_day": 1,
      "gregorian_date": "2021-04-14",
      "note": "Myanmar New Year corresponds to mid-April Gregorian"
    }
  ],

  "conversion_library": "Use 'myanmar-calendar' library for accurate conversions",
  "conversion_notes": "Lunar calendar requires calculation - do not hardcode mappings beyond examples"
}
```

---

## Terminology Database

### Military Terms (`terminology/military_terms.json`)

```json
{
  "military_terms": [
    {
      "term_burmese": "တပ်မတော်",
      "romanization": "tatmadaw",
      "english": "military",
      "context": "General term for Myanmar armed forces"
    },
    {
      "term_burmese": "ပေါ့ပါးခြေလျင်တပ်မ",
      "romanization": "light infantry division",
      "english": "Light Infantry Division",
      "abbreviation": "LID",
      "context": "Major combat unit type"
    },
    {
      "term_burmese": "ခြေလျင်တပ်ရင်း",
      "romanization": "infantry battalion",
      "english": "Infantry Battalion",
      "abbreviation": "IB"
    },
    {
      "term_burmese": "အမှတ်",
      "romanization": "amhto",
      "english": "number",
      "context": "Used in unit designations, e.g., 'LID number 99'"
    },
    {
      "term_burmese": "တပ်မ",
      "romanization": "tapa ma",
      "english": "division"
    }
  ],

  "action_terms": [
    {
      "term_burmese": "ပစ်ခတ်",
      "romanization": "pyitkhut",
      "english": "shoot / shell / fire upon"
    },
    {
      "term_burmese": "တိုက်ခိုက်",
      "romanization": "taikhite",
      "english": "attack"
    },
    {
      "term_burmese": "ဖမ်းဆီး",
      "romanization": "fanzi",
      "english": "arrest / detain"
    },
    {
      "term_burmese": "မီးရှို့",
      "romanization": "mi hsho",
      "english": "burn (with fire)"
    }
  ]
}
```

---

## Database Usage in RISE Tools

### rise-extract: Entity Recognition

```python
# Load context database
with open('rise/contexts/myanmar/military/tatmadaw_units.json') as f:
    units_db = json.load(f)

# Extract military unit from text
text = "လိုင်း ၉၉ အမှတ် ၁၄:၀၀ နာရီတွင် ရွာကို ပစ်ခတ်ခဲ့သည်"
extracted_term = "လိုင်း ၉၉"

# Check aliases
unit_id = check_unit_alias(extracted_term)  # → "tatmadaw_lid_99"

# Lookup in database
unit_data = find_unit_by_id(units_db, unit_id)

# Extract entity with context boost
entity = {
    "entity_id": "ent_mil_001",
    "name_burmese": "လိုင်း ၉၉",
    "name_english": unit_data["official_name_english"],  # "Light Infantry Division 99"
    "unit_type": unit_data["unit_type"],
    "confidence": 0.95,  # High confidence due to database match
    "context_validated": True,
    "database_match": unit_id
}
```

### rise-verify: Validation

```python
# Verify extracted location against geographic database
extracted_location = "Thayetchaung Village, Kale Township"

# Load geography database
villages_db = load_database('rise/contexts/myanmar/geography/villages.json')

# Search
village = search_village(villages_db, "Thayetchaung", "Kale")

if village:
    # Match found - validate and enrich
    verified_location = {
        "name": village["name_english"],
        "township": village["township_id"],
        "coordinates": village["coordinates"],
        "confidence": 0.95,  # Boosted by database match
        "context_validated": True
    }
else:
    # No match - flag for review
    verified_location = {
        "name": extracted_location,
        "confidence": 0.60,  # Lowered - no database confirmation
        "context_validated": False,
        "review_needed": True
    }
```

---

## Database Maintenance

### Updating Context Databases

1. **New units identified**:
   - Add to `tatmadaw_units.json`
   - Update `unit_aliases.json` with common names
   - Document verification level and sources

2. **Geographic changes**:
   - Village name changes (post-conflict renaming)
   - Administrative boundary changes
   - Update with date and reason

3. **Version control**:
   - Increment version number
   - Document changes in CHANGELOG
   - Preserve old versions for reproducibility

### Verification Levels

- **confirmed**: Multiple reliable sources confirm existence/details
- **probable**: Single reliable source or multiple unverified sources
- **unverified**: Mentioned in evidence but not confirmed by independent sources
- **disputed**: Conflicting information about existence/details

---

## Database Sources

Myanmar context databases compiled from:
- Myanmar Peace Monitor (conflict tracking)
- Open-source intelligence (defector reports, leaked documents)
- Tatmadaw organizational charts (when available)
- Myanmar government administrative data (pre-coup)
- OpenStreetMap (geographic data)
- Academic research on Myanmar military
- Field documentation networks

All sources documented in database metadata.

---

*Context databases transform RISE from generic text processing into Myanmar-specific forensic intelligence. Domain knowledge encoded as data.*
