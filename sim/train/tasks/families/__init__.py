"""One module per skill family; each exports ``TASKS`` (a list of TaskDefinition).

Modules here are discovered automatically by ``sim.train.tasks.catalog``; there is
no shared registry to edit. Modules whose name starts with ``_`` are skipped.
"""
