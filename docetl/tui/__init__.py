"""Interactive terminal UI for visualizing DocETL pipeline progress.

The entry point is :func:`docetl.tui.app.run_with_tui`, invoked from
``DSLRunner.load_run_save`` when ``ui: "tui"`` is set in the pipeline config.
For agent-orchestrated runs, ``ui: "web"`` launches the browser-based feedback
UI via :func:`docetl.tui.web_reporter.run_with_web_ui`.
"""
