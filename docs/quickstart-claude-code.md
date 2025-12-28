# Quick Start with Claude Code

The fastest way to build DocETL pipelines is with [Claude Code](https://claude.ai/download), Anthropic's agentic coding tool. DocETL includes a built-in Claude Code skill that helps you create, run, and debug pipelines interactively.

## Option 1: Clone the Repository (Recommended)

This gives you the full development environment with the skill already configured.

1. Follow the [Installation from Source](installation.md#installation-from-source) instructions
2. Run `claude` in the repository directory

The skill is located at `.claude/skills/docetl/SKILL.md`.

## Option 2: Install via pip

If you already have DocETL installed via pip, you can install the skill separately:

```bash
pip install docetl
docetl install-skill
```

This copies the skill to `~/.claude/skills/docetl/`. Then run `claude` in any directory.

To uninstall: `docetl install-skill --uninstall`

## Usage

Simply describe what you want to do with your data. The skill activates automatically when you mention "docetl" or describe unstructured data processing tasks:

```
> I have a folder of customer support tickets in JSON format.
> I want to extract the main issue, sentiment, and suggested resolution for each.
```

Claude will:

1. **Read your data** to understand its structure
2. **Write a tailored pipeline** with prompts specific to your documents
3. **Run the pipeline** and show you the results
4. **Debug issues** if any operations fail

## Alternative: Manual Pipeline Authoring

If you prefer not to use Claude Code, see the [Quick Start Tutorial](tutorial.md) for writing pipelines by hand.
