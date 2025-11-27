# MCP Server

DocETL ships an MCP (Model Context Protocol) server so you can attach it to AI clients (Cursor, ChatGPT Desktop, Claude Desktop) and use tools to:

- Create datasets from local directories
- Author and validate arbitrary DocETL pipelines
- Run pipelines and view outputs
- Browse examples and quick docs from within your client

## Installation

Install DocETL with the MCP and parsing extras:

```bash
pip install "docetl[parsing,mcp]"
```

Recommended: Put your LLM API key in an `.env` file at your project root:

```
OPENAI_API_KEY=your_api_key_here
```

## Start the server

Two equivalent options:

```bash
docetl mcp serve
```

This starts a stdio MCP server compatible with tools like Cursor and ChatGPT Desktop.

## Attach in clients

### Cursor example

Add to your Cursor settings:

```json
{
  "mcpServers": {
    "docetl": {
      "command": "docetl",
      "args": ["mcp", "serve"],
      "env": { "OPENAI_API_KEY": "YOUR_KEY" },
      "cwd": "/abs/path/your-project"
    }
  }
}
```

### Claude Desktop example

Create or edit your Claude Desktop MCP config file:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`
- Windows: `%AppData%/Claude/claude_desktop_config.json`

Add an entry like:

```json
{
  "mcpServers": {
    "docetl": {
      "command": "docetl",
      "args": ["mcp", "serve"],
      "env": { "OPENAI_API_KEY": "YOUR_KEY" },
      "cwd": "/abs/path/your-project"
    }
  }
}
```

Restart Claude Desktop after saving the file.

### What is cwd?

`cwd` is the working directory for the DocETL MCP server process launched by your client.
- Relative paths in your pipeline YAML (e.g., datasets or output files) are resolved against this directory.
- DocETL looks for a `.env` file in `cwd` to load API keys (e.g., `OPENAI_API_KEY`).
- Set `cwd` to the folder where your datasets and pipeline files live.

Example:
- If your files are here:
  - `/Users/you/projects/my-docetl-run/dataset.json`
  - `/Users/you/projects/my-docetl-run/pipeline.yaml`
- Then set:
  - `cwd`: `/Users/you/projects/my-docetl-run`

#### No files yet?

That’s fine. Have your AI agent generate the pipeline YAML inline, then pass it directly to `pipeline.run`. Any relative paths in that YAML (like `datasets.input.path: dataset.json` or `pipeline.output.path: output.json`) will be created under `cwd` at run time. You do not need to create these files in advance.

## What it can do

- Turn a folder of PDFs/TXT/DOCX/PPTX/XLSX into a DocETL dataset
- Draft pipelines from your instructions (any operators: map, reduce, resolve, split/gather, code ops, etc.)
- Validate and run pipelines; write outputs to JSON/CSV
- Show examples and brief docs inside your AI client
- Optional: use Azure Document Intelligence for PDFs (only if you’ve set keys)

## How to use it (in your AI chat)

You don’t need to know any API names. Speak in natural language; the AI will use the MCP server under the hood.

1) Point it at your data
   - “I have documents in /abs/path/data (PDFs and TXTs). Prepare whatever you need to use that folder as a DocETL input. Prefer lazy parsing for PDFs. Use /abs/path/output.json for results.”

2) Ask it to draft a pipeline
   - “Create a minimal pipeline that summarizes each document into a single summary field. Use sensible defaults and make sure prompts reference the document text. Show me the YAML you plan to run.”

3) Have it run and show results
   - “Run the pipeline now, then show me: total cost, output file path, and the first 3 rows.”

4) Iterate
   - “Add a resolve stage to deduplicate by company name, then produce one summary per company. Re-run and show the results.”
   - “Save the final YAML as pipeline.yaml under the working directory, then run it again.”

## Azure usage (optional)

If you have Azure Document Intelligence credentials:

```bash
export DOCUMENTINTELLIGENCE_API_KEY=...
export DOCUMENTINTELLIGENCE_ENDPOINT=...
```

Use `dataset.create_from_directory_azure_di` to create a dataset configured for `azure_di_read`. By default, the server does not use Azure; you must opt into this tool explicitly.


