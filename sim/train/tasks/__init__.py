"""Training task definitions, separate from the held-out validation registry."""

from .catalog import TRAIN_TASKS, load_train_task
from .schema import TaskDefinition, validate_catalog

__all__ = ["TaskDefinition", "TRAIN_TASKS", "load_train_task", "validate_catalog"]
