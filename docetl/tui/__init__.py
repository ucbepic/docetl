"""Interactive terminal UI for visualizing DocETL pipeline progress.

See ``docs/design/progress-visualization.md``. The entry point is
:func:`docetl.tui.app.run_with_tui`, invoked from ``DSLRunner.load_run_save``
when ``pipeline.interactive_ui`` is set (and stdout is a TTY).
"""
