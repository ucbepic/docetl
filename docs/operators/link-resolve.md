# Cluster operation

The `link_resolve` operation in DocETL is used to fix links between
items, e.g. in a knoweldge graph. It assumes you have already ensured
that the item id:s themselves are canonical, e.g. by running resolve
first.

It will examine every id specified in a link from one item to another
and compare it to all item id:s. If it is not present with an exact
match, it is going to be compared to each of them using an llm prompt
to try to find a match.

Note that this is a one-sided approach, compared to the resolve
operation: It assumes that item id:s are canonical / correct.

## 🚀 Example: Knowledge graph of boating terms

```yaml
- name: fix_links
  type: link_resolve
  id_key: title
  link_key: related_to
  blocking_threshold: 0.85
  embedding_model: text-embedding-ada-002
  comparison_model: gpt-4o-mini
  comparison_prompt: |
    Compare the following two concepts:

    Concept 1: [{{ link_value }}]
    Concept 2: [{{ id_value }}]

    Are these concepts likely refering to the same thing? When
    comparing them, also consider the following description of
    concept 2:

      {{ item.description }}

    Respond with "True" if they are likely the same concept, or "False" if they are likely different concepts.
```

??? example "Sample Input and Output"

The above make two replacements in the `related_to` keys: `Main sail` -> `Sail (main)` and `Sailing boat` -> `Sailing vessel`.

    Input:
    ```json
    [
      {
        "title": "Sailing vessel",
        "related_to": ["Main sail", "Main mast", "Rudder"],
        "description": "A boat or ship propelled by sails. Typically with a very hydrodynamical hull."
      },
      {
        "title": "Catamaran",
        "related_to": ["Sailing boat", "Hull"],
        "description": "A type of vessel with two parallel hulls"
      },
      {
        "title": "Sail (main)",
        "related_to": ["Sheet"],
        "description": "A cloth set up to catch the force of the wind"
      },
      {
        "title": "Sheet (sailing)",
        "related_to": [],
        "description": "Ropes used to trim the angle of the sails, deciding the (angle of) wind force applied to the masts"
      },
      {
        "title": "Rudder angle",
        "related_to": [],
        "description": "The rudder angle together with the wind force on the mast(s) decides the rate of turn of the vessel"
      }
    ]
    ```

    Output:
    ```json
    [
      {
        "title": "Sailing vessel",
        "related_to": ["Sail (main)", "Main mast", "Rudder"],
        "description": "A boat or ship propelled by sails. Typically with a very hydrodynamical hull."
      },
      {
        "title": "Catamaran",
        "related_to": ["Sailing vessel", "Hull"],
        "description": "A type of vessel with two parallel hulls"
      },
      {
        "title": "Sail (main)",
        "related_to": ["Sheet"],
        "description": "A cloth set up to catch the force of the wind"
      },
      {
        "title": "Sheet (sailing)",
        "related_to": [],
        "description": "Ropes used to trim the angle of the sails, deciding the (angle of) wind force applied to the masts"
      },
      {
        "title": "Rudder angle",
        "related_to": [],
        "description": "The rudder angle together with the wind force on the mast(s) decides the rate of turn of the vessel"
      }
    ]
    ```

## Required Parameters

- `name`: A unique name for the operation.
- `type`: Must be set to "link_resolve".
- `id_key`: A key to use for item id:s
- `link_key`: A key to make replacements in
- `blocking_threshold`: Embedding similarity threshold for considering entries as potential matches
- `comparison_prompt`: The prompt template to use for comparing potential matches.

## Optional Parameters
