# Interactive Progress View

DocETL can show a full-screen, live progress dashboard in your terminal while a
pipeline runs. It lets you watch each document complete, see cost and timing per
operation, and click into any finished document to inspect its output, the
prompt that produced it, and where it came from.

![The progress view: steps and operations on the left, a grid of documents in
the middle, and the selected document's detail on the right](../assets/progress-view/tui-real-complete.png)

## Turning it on

First install the optional `tui` extra (it pulls in the
[Textual](https://textual.textualize.io/) library used to draw the dashboard):

```bash
pip install "docetl[tui]"
```

Then add `interactive_ui: true` at the top level of your config (next to
`default_model`):

```yaml
default_model: gpt-4.1-nano
interactive_ui: true

pipeline:
  steps:
    - name: themes
      input: reviews
      operations:
        - extract_theme
        - canonicalize_themes
        - summarize_themes
  output:
    type: file
    path: output.json
```

And run the pipeline the usual way:

```bash
docetl run pipeline.yaml
```

The dashboard only starts when you are in an interactive terminal. In a script,
a CI job, or anywhere the output is piped, DocETL automatically falls back to its
normal log output, so the flag is safe to leave on.

## What you see

There are three panels:

- **Left — operations.** Every step and operation in your pipeline, with a live
  status, the running counts, the cost, and the elapsed time. The total cost and
  time for the whole run are at the top.
- **Middle — documents.** One circle per document for the selected operation. A
  filled circle means the document is done (green), in progress (orange), or
  errored (red); a hollow circle means it has not started yet. The header shows
  how far along the operation is.
- **Right — detail.** The output of the document under the cursor, the prompt
  that produced it, and a short note about where it came from. Documents appear
  here as soon as they finish, so you can inspect results while the run is still
  going.

## Moving around

| Key | Action |
| --- | --- |
| `↑` / `↓` | select an operation |
| `Tab` | switch between the operations list and the document grid |
| `←` / `→` | move the cursor in the grid (and page through large grids) |
| `PgUp` / `PgDn` | page through the grid |
| `Enter` | inspect the document under the cursor |
| `q` | quit (the run keeps going) |

## What each kind of operation shows

The progress bar counts the right thing for each operation, and the detail panel
adds a short note about where a document came from when that makes sense.

**Reduce** counts groups, and a group shows how many documents were combined into
it:

![Reduce showing groups and source counts](../assets/progress-view/tui-reduce-groups.png)

**Split** counts chunks, and a chunk shows which piece of its source document it
is (for example, "chunk 2 of 5"):

![Split showing chunks](../assets/progress-view/tui-split-chunks.png)

**Resolve** and **equijoin** count comparisons as they are made. Their output
documents aren't known until the operation finishes, so while it runs the header
counts comparisons and the grid shows a `?`:

![Resolve showing comparisons](../assets/progress-view/tui-resolve-comparisons.png)

## Very large runs

For runs with tens of thousands of documents the grid would never fit on screen,
so it switches to a heatmap: each cell stands for a bucket of documents, shaded
by how many are done. The live counts and per-operation totals still tell you
exactly where things stand.

![Heatmap view for a large run](../assets/progress-view/tui-scale-heatmap.png)
