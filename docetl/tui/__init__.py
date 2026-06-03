"""Interactive terminal UI for visualizing DocETL pipeline progress.

The entry point is :func:`docetl.tui.app.run_with_tui`, invoked from
``DSLRunner.load_run_save`` when the top-level ``interactive_ui`` flag is set
(and stdout is a TTY).
"""
