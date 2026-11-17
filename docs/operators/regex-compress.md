# Regex Compress Operation

The Regex Compress operation cuts a long document down to the paragraphs that matter before an LLM reads it, with no LLM call. Use it in front of a map, extract, or filter when only some paragraphs of each document matter and keywords find them, such as the clauses of a contract.

## Example

```yaml
- name: keep_clause_paragraphs
  type: regex_compress
  document_key: contract
  keep_patterns:
    - "(?i)\\bgovern(ed|ing)\\s+law\\b"
    - "(?i)\\bnon-?compet"
    - "(?i)\\b(terminat(e|ion)|renew(al)?)\\b"
  drop_patterns:
    - "(?i)^\\s*table of contents"
  remove_patterns:
    - "(?m)^\\s*Page \\d+( of \\d+)?\\s*$"
  keep_first_chars: 1500
```

## How it works

1. Delete every match of each `remove_patterns` entry, such as page numbers or repeated headers.
2. Split the text into paragraphs at blank lines, and drop empty paragraphs.
3. Keep a paragraph when no `drop_patterns` entry matches it, and either the paragraph starts within the first `keep_first_chars` characters or a `keep_patterns` entry matches it.
4. Join the kept paragraphs with blank lines, with `gap_marker` (`[...]` by default) where paragraphs were left out.

When no paragraph is kept, the field keeps the whole text from step 1.

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `document_key` | The field to compress | Required |
| `output_key` | The field to write | `document_key` |
| `keep_patterns` | A paragraph that matches any of these is kept | `[]` |
| `drop_patterns` | A paragraph that matches any of these is dropped, even when a keep pattern matches | `[]` |
| `remove_patterns` | Matches are deleted before the split | `[]` |
| `keep_first_chars` | Paragraphs that start within this many characters are kept | `0` |
| `gap_marker` | Text between kept paragraphs that were not next to each other; `""` for none | `[...]` |

Patterns are searched anywhere in a paragraph. Use syntax that both Python's `re` and Rust's `regex` accept: inline flags such as `(?i)` and `(?m)`, character classes, `\b`, `\s`, `\d`, alternation, and groups. Lookahead, lookbehind, named groups, and backreferences are refused, because the [Quail engine](../execution/quail.md) runs the operation as SQL with Rust's `regex`.
